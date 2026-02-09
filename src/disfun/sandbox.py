"""Fork-based sandbox for executing LLM-generated code.

Uses os.fork() for process isolation. Child inherits parent's memory (copy-on-write),
so cached graphs are available without reload. Child runs evaluation, writes result
to pipe, exits. Parent waits with timeout.

This is much faster than subprocess-based sandboxing because:
1. Fork is ~1ms vs subprocess ~50ms
2. Child already has graphs in memory (no LMDB reload)
3. No pickle serialization of function or input needed
"""

import ast
import os
import signal
import resource
import time
import cloudpickle
import threading


def cleanup_orphaned_sandbox_processes(logger=None, max_age_seconds=300):
    """No-op. Fork-based sandbox children are reaped by parent via waitpid."""
    return 0


class ExternalProcessSandbox:
    """Sandbox using fork() for fast, isolated execution.

    Child process inherits parent's memory via copy-on-write, so graphs
    cached in the evaluator are available without reloading.
    """

    # Cache for compiled base namespace (spec without priority), per process
    _cached_namespace = None
    _cached_base_hash = None
    _compile_lock = threading.Lock()  # Thread synchronization for cache

    def __init__(
        self,
        timeout_secs: int = 30,
        graph_dir: str = None,
        memory_limit_gb: float = 1.0,
        debug: bool = False,
    ):
        self.timeout_secs = timeout_secs
        self.graph_dir = graph_dir
        self.memory_limit_gb = memory_limit_gb
        self.debug = debug

    @staticmethod
    def compile_code(program: str):
        """Compile program with caching. Only recompiles priority function.

        Separates the program into base (imports, helpers) and priority function.
        The base is cached and reused across evaluations.
        Thread-safe: uses lock to prevent race conditions in cache access.
        """
        tree = ast.parse(program)

        # Separate priority from rest of program
        priority_node = None
        base_nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'priority':
                priority_node = node
            else:
                base_nodes.append(node)

        # Check if base changed (first call or different spec)
        base_tree = ast.Module(body=base_nodes, type_ignores=[])
        base_hash = hash(ast.dump(base_tree))

        # Thread-safe cache check and update
        with ExternalProcessSandbox._compile_lock:
            if ExternalProcessSandbox._cached_base_hash != base_hash:
                ExternalProcessSandbox._cached_namespace = {}
                exec(compile(base_tree, '<ast>', 'exec'), ExternalProcessSandbox._cached_namespace)
                ExternalProcessSandbox._cached_base_hash = base_hash

                # Pre-load lazy modules so child inherits them (avoids mmap in sandbox)
                # numpy.random is lazy-loaded and uses ~2GB VMS when mmap'd fresh
                ns = ExternalProcessSandbox._cached_namespace
                if 'np' in ns or 'numpy' in ns:
                    np_module = ns.get('np') or ns.get('numpy')
                    if np_module is not None:
                        try:
                            _ = np_module.random.rand(1)  # Force numpy.random load
                        except Exception:
                            pass

            # Always compile and inject new priority into cached namespace
            if priority_node:
                priority_tree = ast.Module(body=[priority_node], type_ignores=[])
                exec(compile(priority_tree, '<ast>', 'exec'), ExternalProcessSandbox._cached_namespace)

            return ExternalProcessSandbox._cached_namespace.copy()  # Return copy to avoid mutation

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input,
    ) -> tuple:
        """Execute function in fork-based sandbox.

        Child inherits parent's memory (including cached graphs) via copy-on-write.
        Sets memory limits in child, runs evaluation, returns result via pipe.

        Returns:
            (result, success, cpu_time)
        """
        try:
            # Compile in parent - child will inherit compiled namespace
            namespace = ExternalProcessSandbox.compile_code(program)
            func = namespace[function_to_run]
        except Exception as e:
            if self.debug:
                import traceback
                import sys
                sys.stderr.write(f"SANDBOX COMPILE ERROR: {type(e).__name__}: {e}\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return None, False, 0.0

        # Create pipe for result communication
        read_fd, write_fd = os.pipe()

        pid = os.fork()

        if pid == 0:
            # === CHILD PROCESS ===
            os.close(read_fd)

            # Suppress stdout/stderr from LLM-generated code (e.g. print statements)
            # to prevent unbounded growth of SBATCH .out/.err files.
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)  # stdout
            if not self.debug:
                os.dup2(devnull, 2)  # stderr (keep if debug mode)
            os.close(devnull)

            try:
                # Set memory limit (virtual address space)
                # Note: Use RLIMIT_AS, not RLIMIT_DATA - the latter breaks mmap
                # for shared libraries like numpy ("failed to map segment")
                mem_bytes = int(self.memory_limit_gb * 1024 * 1024 * 1024)
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, resource.error):
                    pass

                # Run evaluation
                start_cpu = time.process_time()
                result = func(test_input, self.graph_dir)
                cpu_time = time.process_time() - start_cpu

                # Serialize and write result
                result_bytes = cloudpickle.dumps({
                    "result": result,
                    "cpu_time": cpu_time
                })
                os.write(write_fd, result_bytes)
                os.close(write_fd)
                os._exit(0)

            except Exception as e:
                if self.debug:
                    import traceback
                    import sys
                    sys.stderr.write(f"SANDBOX ERROR: {type(e).__name__}: {e}\n")
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                os._exit(1)

        else:
            # === PARENT PROCESS ===
            os.close(write_fd)

            try:
                # Wait for child with timeout
                deadline = time.time() + self.timeout_secs
                child_done = False

                while time.time() < deadline:
                    pid_result, status = os.waitpid(pid, os.WNOHANG)
                    if pid_result != 0:
                        child_done = True
                        break
                    time.sleep(0.01)

                if not child_done:
                    # Timeout - kill child
                    try:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                    except (ProcessLookupError, ChildProcessError):
                        pass
                    os.close(read_fd)
                    return None, False, 0.0

                # Check exit status
                if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0):
                    os.close(read_fd)
                    return None, False, 0.0

                # Read result from pipe
                result_chunks = []
                while True:
                    chunk = os.read(read_fd, 65536)
                    if not chunk:
                        break
                    result_chunks.append(chunk)
                os.close(read_fd)

                if not result_chunks:
                    return None, False, 0.0

                result_bytes = b''.join(result_chunks)
                result_data = cloudpickle.loads(result_bytes)
                return result_data["result"], True, result_data["cpu_time"]

            except Exception:
                # Clean up child if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
                try:
                    os.close(read_fd)
                except OSError:
                    pass
                return None, False, 0.0

    def cleanup_all(self):
        """No-op. Fork-based sandbox doesn't create files."""
        pass
