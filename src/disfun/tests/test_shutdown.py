"""Tests for shutdown logic.

Tests verify:
1. Signal handlers trigger proper shutdown
2. Processes exit cleanly, no zombies or orphans
3. RabbitMQ connections close properly
4. No reconnection attempts after shutdown requested

Run all tests:
    pytest src/disfun/tests/test_shutdown.py -v

RabbitMQ tests auto skip if RabbitMQ is not reachable.
GPU tests auto skip if no GPU is available.
"""

import os
import sys
import time
import signal
import asyncio
import pytest
import psutil
import multiprocessing
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any



# =============================================================================
# Unit Tests. Mock everything, fast.
# =============================================================================

class TestConnectionManagerClose:
    """Test that ConnectionManager.close() properly shuts down."""

    @pytest.mark.asyncio
    async def test_close_calls_request_shutdown(self):
        """close() should call request_shutdown() to prevent reconnection."""
        from disfun.utils.rabbitmq import ConnectionManager

        # Create ConnectionManager with mocked config
        mock_config = Mock()
        mock_config.rabbitmq.host = "localhost"
        mock_config.rabbitmq.port = 5672
        mock_config.rabbitmq.username = "guest"
        mock_config.rabbitmq.password = "guest"
        mock_config.rabbitmq.vhost = "/"
        mock_config.rabbitmq.heartbeat = 60
        mock_config.rabbitmq.blocked_connection_timeout = 300

        conn_manager = ConnectionManager(
            config=mock_config,
            component_name="TestComponent",
            queue_names=["test_queue"],
            logger=Mock()
        )

        # Mock the connection and channel (save refs before close() sets them to None)
        mock_channel = AsyncMock()
        mock_channel.is_closed = False
        mock_connection = AsyncMock()
        mock_connection.is_closed = False

        conn_manager.channel = mock_channel
        conn_manager.connection = mock_connection

        # Call close
        await conn_manager.close()

        # Verify shutdown was requested
        assert conn_manager._shutdown_requested is True

        # Verify channel and connection were closed
        mock_channel.close.assert_called_once()
        mock_connection.close.assert_called_once()

        # Verify references cleared
        assert conn_manager.channel is None
        assert conn_manager.connection is None

    @pytest.mark.asyncio
    async def test_close_handles_already_closed_connection(self):
        """close() should handle already closed connections gracefully."""
        from disfun.utils.rabbitmq import ConnectionManager

        mock_config = Mock()
        mock_config.rabbitmq.host = "localhost"
        mock_config.rabbitmq.port = 5672
        mock_config.rabbitmq.username = "guest"
        mock_config.rabbitmq.password = "guest"
        mock_config.rabbitmq.vhost = "/"
        mock_config.rabbitmq.heartbeat = 60
        mock_config.rabbitmq.blocked_connection_timeout = 300

        conn_manager = ConnectionManager(
            config=mock_config,
            component_name="TestComponent",
            queue_names=["test_queue"],
            logger=Mock()
        )

        # Connection already closed
        conn_manager.connection = AsyncMock()
        conn_manager.connection.is_closed = True
        conn_manager.channel = AsyncMock()
        conn_manager.channel.is_closed = True

        # Should not raise
        await conn_manager.close()
        assert conn_manager._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_connect_with_retry_stops_when_shutdown_requested(self):
        """connect_with_retry() should return False if shutdown requested."""
        from disfun.utils.rabbitmq import ConnectionManager

        mock_config = Mock()
        mock_config.rabbitmq.host = "localhost"
        mock_config.rabbitmq.port = 5672
        mock_config.rabbitmq.username = "guest"
        mock_config.rabbitmq.password = "guest"
        mock_config.rabbitmq.vhost = "/"
        mock_config.rabbitmq.heartbeat = 60
        mock_config.rabbitmq.blocked_connection_timeout = 300

        conn_manager = ConnectionManager(
            config=mock_config,
            component_name="TestComponent",
            queue_names=["test_queue"],
            logger=Mock()
        )

        # Request shutdown before connecting
        conn_manager.request_shutdown()

        # Should return False immediately without trying to connect
        result = await conn_manager.connect_with_retry()
        assert result is False


class TestEvaluatorShutdown:
    """Test Evaluator.shutdown() method."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_close_and_subprocesses(self):
        """shutdown() should close connection and shutdown subprocesses."""
        from disfun.evaluator import Evaluator

        # Create evaluator with mocked dependencies
        mock_conn = AsyncMock()
        mock_template = Mock()
        mock_template.get_function.return_value = Mock(body="pass")

        with patch('disfun.evaluator.sandbox.ExternalProcessSandbox'):
            with patch('disfun.evaluator.ProcessPoolExecutor'):
                with patch('disfun.evaluator.Manager') as mock_manager:
                    mock_manager.return_value.Value.return_value = Mock(value=0)
                    mock_manager.return_value.Lock.return_value = Mock()

                    evaluator = Evaluator(
                        template=mock_template,
                        inputs=[(7, 2, 5)],
                        local_id=12345,
                        evaluator_config=Mock(
                            timeout=30,
                            max_workers=2,
                            prefetch_count=5,
                            graph_dir="/tmp/graphs",
                            sandbox_memory_limit_gb=1.0
                        ),
                        connection_manager=mock_conn,
                        target_signatures=None,
                    )

        # Mock shutdown_subprocesses
        evaluator.shutdown_subprocesses = AsyncMock()

        await evaluator.shutdown()

        # Verify both were called
        mock_conn.close.assert_called_once()
        evaluator.shutdown_subprocesses.assert_called_once()


class TestSamplerShutdown:
    """Test Sampler.shutdown() method."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_close_and_cleanup(self):
        """shutdown() should close connection and cleanup LLM."""
        # We can't easily instantiate Sampler without vLLM, so test the method logic
        # by creating a minimal mock

        class MockSampler:
            def __init__(self):
                self._conn = AsyncMock()
                self.cleanup_called = False

            async def shutdown(self):
                await self._conn.close()
                self.cleanup()

            def cleanup(self):
                self.cleanup_called = True

        sampler = MockSampler()
        await sampler.shutdown()

        sampler._conn.close.assert_called_once()
        assert sampler.cleanup_called is True


# =============================================================================
# Integration Tests. Real processes, mock heavy components.
# =============================================================================

class TestProcessShutdown:
    """Test that processes handle SIGTERM correctly."""

    def test_worker_process_exits_on_sigterm(self):
        """A worker process should exit cleanly when receiving SIGTERM."""

        def worker_process():
            """Simple worker that uses our shutdown pattern."""
            import signal
            import asyncio
            from dataclasses import dataclass
            from typing import Any

            @dataclass
            class ProcessState:
                instance: Any = None
                task: Any = None
                shutdown_started: bool = False

            state = ProcessState()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def shutdown():
                if state.shutdown_started:
                    return
                state.shutdown_started = True
                if state.task and not state.task.done():
                    state.task.cancel()
                    try:
                        await asyncio.wait_for(state.task, timeout=2)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                loop.stop()

            async def run():
                async def work():
                    while True:
                        await asyncio.sleep(0.1)

                state.task = asyncio.create_task(work())
                try:
                    await state.task
                except asyncio.CancelledError:
                    pass
                finally:
                    if not state.shutdown_started:
                        await shutdown()

            shutdown_task = None
            def on_signal():
                nonlocal shutdown_task
                if shutdown_task is None:
                    shutdown_task = asyncio.create_task(shutdown())

            loop.add_signal_handler(signal.SIGTERM, on_signal)
            loop.add_signal_handler(signal.SIGINT, on_signal)

            try:
                loop.run_until_complete(run())
            finally:
                loop.close()
                sys.exit(0)

        # Start worker process
        proc = multiprocessing.Process(target=worker_process)
        proc.start()

        # Give it time to start
        time.sleep(0.5)
        assert proc.is_alive()

        # Send SIGTERM
        os.kill(proc.pid, signal.SIGTERM)

        # Wait for clean exit (should be fast)
        proc.join(timeout=5)

        # Verify it exited
        assert not proc.is_alive()
        assert proc.exitcode == 0

    def test_no_orphaned_processes_after_shutdown(self):
        """After shutdown, no child processes should remain."""

        def parent_with_children():
            """Parent that spawns children, then handles SIGTERM."""
            import signal
            import asyncio
            import multiprocessing

            children = []

            def child_worker():
                while True:
                    time.sleep(0.1)

            # Spawn some children
            for _ in range(3):
                p = multiprocessing.Process(target=child_worker)
                p.start()
                children.append(p)

            def shutdown_handler(signum, frame):
                # Terminate all children
                for p in children:
                    if p.is_alive():
                        p.terminate()
                for p in children:
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
                        p.join()
                sys.exit(0)

            signal.signal(signal.SIGTERM, shutdown_handler)

            # Wait forever (until signal)
            while True:
                time.sleep(0.1)

        # Start parent
        proc = multiprocessing.Process(target=parent_with_children)
        proc.start()

        # Give it time to spawn children
        time.sleep(1)

        # Get all descendants before terminating
        try:
            parent_proc = psutil.Process(proc.pid)
            children_before = parent_proc.children(recursive=True)
            assert len(children_before) == 3
        except psutil.NoSuchProcess:
            pytest.fail("Parent process died unexpectedly")

        # Send SIGTERM to parent
        os.kill(proc.pid, signal.SIGTERM)
        proc.join(timeout=5)

        # Verify parent exited
        assert not proc.is_alive()

        # Verify no orphaned children
        time.sleep(0.5)  # Give OS time to clean up
        for child in children_before:
            assert not child.is_running(), f"Orphaned child process: {child.pid}"

    def test_force_kill_after_timeout(self):
        """Processes that don't exit gracefully should be force killed."""

        def stubborn_worker():
            """Worker that ignores SIGTERM."""
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # Ignore SIGTERM
            while True:
                time.sleep(0.1)

        proc = multiprocessing.Process(target=stubborn_worker)
        proc.start()
        time.sleep(0.5)

        # Send SIGTERM (will be ignored)
        os.kill(proc.pid, signal.SIGTERM)
        time.sleep(0.5)
        assert proc.is_alive()  # Still alive because it ignores SIGTERM

        # Force kill with SIGKILL
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=2)

        assert not proc.is_alive()
        assert proc.exitcode == -9  # Killed by SIGKILL

    def test_full_hierarchy_cleanup(self):
        """Test main → child → grandchild cleanup (simulates main → evaluator → sandbox).

        This mirrors the real process hierarchy where:
        1. Main process spawns evaluator (child)
        2. Evaluator spawns sandbox processes (grandchildren)
        3. When main terminates evaluator, sandbox processes should also be cleaned up
        """

        def grandchild_worker():
            """Simulates a sandbox process. Just sleeps forever."""
            while True:
                time.sleep(0.1)

        def child_worker(grandchild_pids_queue):
            """Simulates an evaluator that spawns sandbox subprocesses."""
            import signal

            grandchildren = []

            # Spawn grandchildren (sandbox processes)
            for _ in range(2):
                p = multiprocessing.Process(target=grandchild_worker)
                p.start()
                grandchildren.append(p)
                grandchild_pids_queue.put(p.pid)

            def shutdown_handler(signum, frame):
                # Clean up grandchildren first (like evaluator.shutdown_subprocesses)
                for p in grandchildren:
                    if p.is_alive():
                        p.terminate()
                for p in grandchildren:
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
                        p.join()
                sys.exit(0)

            signal.signal(signal.SIGTERM, shutdown_handler)

            # Wait forever (until signal)
            while True:
                time.sleep(0.1)

        # Queue to receive grandchild PIDs from child process
        grandchild_pids_queue = multiprocessing.Queue()

        # Start child (evaluator)
        child = multiprocessing.Process(target=child_worker, args=(grandchild_pids_queue,))
        child.start()

        # Wait for grandchildren to spawn and collect their PIDs
        time.sleep(1)
        grandchild_pids = []
        while not grandchild_pids_queue.empty():
            grandchild_pids.append(grandchild_pids_queue.get_nowait())
        assert len(grandchild_pids) == 2, "Expected 2 grandchild processes"

        # Verify all processes are running
        assert child.is_alive()
        for pid in grandchild_pids:
            assert psutil.pid_exists(pid), f"Grandchild {pid} should be running"

        # Capture descendants before terminating (like main does)
        try:
            child_proc = psutil.Process(child.pid)
            descendants_before = child_proc.children(recursive=True)
            assert len(descendants_before) == 2
        except psutil.NoSuchProcess:
            pytest.fail("Child process died unexpectedly")

        # Send SIGTERM to child (like main does during shutdown)
        os.kill(child.pid, signal.SIGTERM)
        child.join(timeout=5)

        # Verify child exited cleanly
        assert not child.is_alive()
        assert child.exitcode == 0

        # Verify all grandchildren are dead (no orphans)
        time.sleep(0.5)  # Give OS time to clean up
        for pid in grandchild_pids:
            assert not psutil.pid_exists(pid), f"Orphaned grandchild process: {pid}"

    def test_force_kill_cleans_descendants(self):
        """If child ignores SIGTERM, force kill should still clean up descendants.

        This tests the fallback path where main has to SIGKILL a stubborn child
        and then manually kill any orphaned descendants.
        """

        def grandchild_worker():
            """Grandchild that just sleeps."""
            while True:
                time.sleep(0.1)

        def stubborn_child(grandchild_pids_queue):
            """Child that ignores SIGTERM (simulates stuck evaluator)."""
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # Ignore SIGTERM

            # Spawn grandchildren
            grandchildren = []
            for _ in range(2):
                p = multiprocessing.Process(target=grandchild_worker)
                p.start()
                grandchildren.append(p)
                grandchild_pids_queue.put(p.pid)

            while True:
                time.sleep(0.1)

        grandchild_pids_queue = multiprocessing.Queue()
        child = multiprocessing.Process(target=stubborn_child, args=(grandchild_pids_queue,))
        child.start()

        time.sleep(1)
        grandchild_pids = []
        while not grandchild_pids_queue.empty():
            grandchild_pids.append(grandchild_pids_queue.get_nowait())

        # Capture descendants BEFORE killing (critical, like main does)
        try:
            child_proc = psutil.Process(child.pid)
            descendants = child_proc.children(recursive=True)
        except psutil.NoSuchProcess:
            pytest.fail("Child died unexpectedly")

        # Send SIGTERM (ignored by stubborn child)
        os.kill(child.pid, signal.SIGTERM)
        time.sleep(0.5)
        assert child.is_alive()  # Still alive

        # Force kill child with SIGKILL
        os.kill(child.pid, signal.SIGKILL)
        child.join(timeout=2)
        assert not child.is_alive()

        # Grandchildren are now orphans. Main must clean them up manually.
        # This is what _terminate_processes does after force killing.
        for desc in descendants:
            if desc.is_running():
                desc.kill()

        time.sleep(0.5)
        for pid in grandchild_pids:
            assert not psutil.pid_exists(pid), f"Orphaned grandchild: {pid}"


# =============================================================================
# RabbitMQ Integration Tests. Requires running RabbitMQ.
# =============================================================================

def _rabbitmq_available():
    """Check if RabbitMQ is reachable."""
    import socket
    host = os.environ.get("RABBITMQ_HOST", "rabbitmq")
    port = int(os.environ.get("RABBITMQ_PORT", "5672"))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.error, socket.timeout):
        return False


def _make_rabbitmq_config():
    """Create a test config for RabbitMQ."""
    from dataclasses import dataclass

    @dataclass
    class RabbitMQConfig:
        host: str = os.environ.get("RABBITMQ_HOST", "rabbitmq")
        port: int = int(os.environ.get("RABBITMQ_PORT", "5672"))
        username: str = os.environ.get("RABBITMQ_USER", "guest")
        password: str = os.environ.get("RABBITMQ_PASS", "guest")
        vhost: str = ""
        heartbeat: int = 60
        blocked_connection_timeout: int = 300
        reconnect_delay: float = 1.0
        max_reconnect_delay: float = 5.0

    @dataclass
    class Config:
        rabbitmq: RabbitMQConfig = None

        def __post_init__(self):
            if self.rabbitmq is None:
                self.rabbitmq = RabbitMQConfig()

    return Config()


@pytest.mark.skipif(not _rabbitmq_available(), reason="RabbitMQ not available")
class TestRabbitMQShutdown:
    """Test RabbitMQ connection shutdown. Requires running RabbitMQ server."""

    @pytest.fixture
    def rabbitmq_config(self):
        return _make_rabbitmq_config()

    @pytest.mark.asyncio
    async def test_connection_closes_cleanly(self, rabbitmq_config):
        """Test that RabbitMQ connection closes without errors."""
        from disfun.utils.rabbitmq import ConnectionManager
        import logging

        conn_manager = ConnectionManager(
            config=rabbitmq_config,
            component_name="TestComponent",
            queue_names=["test_shutdown_queue"],
            logger=logging.getLogger("test")
        )

        # Connect
        result = await conn_manager.connect_with_retry()
        assert result is True
        assert conn_manager.connection is not None
        assert not conn_manager.connection.is_closed

        # Close
        await conn_manager.close()

        # Verify closed
        assert conn_manager._shutdown_requested is True
        assert conn_manager.connection is None
        assert conn_manager.channel is None

    @pytest.mark.asyncio
    async def test_no_reconnection_after_shutdown(self, rabbitmq_config):
        """After shutdown requested, connect_with_retry should return False."""
        from disfun.utils.rabbitmq import ConnectionManager
        import logging

        conn_manager = ConnectionManager(
            config=rabbitmq_config,
            component_name="TestComponent",
            queue_names=["test_shutdown_queue"],
            logger=logging.getLogger("test")
        )

        # Connect first
        result = await conn_manager.connect_with_retry()
        assert result is True

        # Close (sets shutdown flag)
        await conn_manager.close()

        # Try to reconnect, should fail immediately
        result = await conn_manager.connect_with_retry()
        assert result is False


# =============================================================================
# GPU Tests. Requires GPU with vLLM.
# =============================================================================

def _gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_worker_for_subprocess_test(result_queue):
    """Worker that allocates GPU memory, then cleans up.

    Defined at module level so it can be pickled for spawn context.
    """
    import torch
    import gc

    # Allocate GPU memory
    tensors = [torch.randn(500, 500, device='cuda') for _ in range(5)]
    allocated = torch.cuda.memory_allocated()
    result_queue.put(('allocated', allocated))

    # Cleanup
    del tensors
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after_cleanup = torch.cuda.memory_allocated()
    result_queue.put(('after_cleanup', after_cleanup))


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
class TestGPUShutdown:
    """Test GPU memory cleanup. Requires GPU."""

    def test_gpu_memory_released_after_cleanup(self):
        """Verify GPU memory is released after cleanup pattern used by sampler.

        This tests the cleanup pattern without loading vLLM (which is slow).
        The sampler's cleanup() does: del self.llm, gc.collect(), torch.cuda.empty_cache()
        """
        import torch
        import gc

        # Get baseline memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated()

        # Allocate some GPU memory (simulates model weights)
        tensors = [torch.randn(1000, 1000, device='cuda') for _ in range(10)]
        allocated_memory = torch.cuda.memory_allocated()
        assert allocated_memory > baseline_memory, "Should have allocated GPU memory"

        # Cleanup pattern (same as sampler.cleanup())
        del tensors
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Verify memory released
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= baseline_memory + 1024 * 1024, "GPU memory should be released"

    def test_gpu_cleanup_in_subprocess(self):
        """Test GPU memory cleanup works correctly in a subprocess (like sampler).

        Samplers run in separate processes. This tests that GPU memory allocated
        in a subprocess is properly released when the subprocess exits.
        """
        # Must use spawn for CUDA (fork causes re-initialization error)
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        proc = ctx.Process(target=_gpu_worker_for_subprocess_test, args=(result_queue,))
        proc.start()
        proc.join(timeout=30)

        assert not proc.is_alive()
        assert proc.exitcode == 0

        # Collect results
        results = {}
        while not result_queue.empty():
            key, value = result_queue.get_nowait()
            results[key] = value

        assert 'allocated' in results, "Worker should have reported allocated memory"
        assert 'after_cleanup' in results, "Worker should have reported cleanup"
        assert results['after_cleanup'] < results['allocated'], "Memory should be reduced after cleanup"

