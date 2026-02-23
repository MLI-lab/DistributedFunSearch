"""Entry points for spawned Sampler and Evaluator processes.

Called by __main__.py via multiprocessing. Must be in a separate module (not __main__)
to be pickle-able with spawn context. Each entry point reloads config and initializes
logging since objects cannot be shared across spawned processes.
"""

import os
import re
import sys
import json
import signal
import asyncio
import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from logging.handlers import RotatingFileHandler
from multiprocessing import current_process


class _NFSSafeRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that suppresses stale file handle errors on NFS.

    When multiple processes share a log file on NFS, rotation by one process
    causes Errno 116 (Stale file handle) in others. The log data is still
    written correctly — only the error traceback to stderr is problematic,
    as it can produce GB-sized .err files on HPC clusters.
    """

    def handleError(self, record):
        """Suppress Errno 116 (Stale file handle) from NFS rotation races."""
        import errno
        _, exc_value, _ = sys.exc_info()
        if isinstance(exc_value, OSError) and exc_value.errno in (
            errno.ESTALE,  # 116 - Stale file handle (NFS)
            errno.ENOENT,  # 2 - File not found (rotation race)
        ):
            # Silently re-open the file on next write attempt
            try:
                self.close()
                self.stream = self._open()
            except Exception:
                pass
            return
        super().handleError(record)


@dataclass
class ProcessState:
    """Mutable state for a worker process. Avoids nonlocal and dict hacks."""
    instance: Any = None
    task: Any = None
    shutdown_started: bool = False


def load_config(config_path):
    """Dynamically load a configuration from a given file path.

    Supports:
        .py files: Dynamically loaded as Python modules (must define Config class)
        .pkl files: Loaded via cloudpickle (for sweep configs with modified params)
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    if config_path.endswith('.pkl'):
        import cloudpickle
        with open(config_path, 'rb') as f:
            return cloudpickle.load(f)

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "Config"):
        raise ValueError(f"The configuration file at {config_path} must define a 'Config' class.")

    return config_module.Config()


def create_logger(name, log_dir, log_file, level=logging.DEBUG,
                   max_bytes=50*1024*1024, backup_count=5,
                   fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   clear_handlers=False):
    """Create a logger with consistent rotating file configuration.

    Args:
        name: Logger name (e.g., 'main_logger', 'resource_logger_123')
        log_dir: Directory for log files
        log_file: Log filename
        level: Logging level (default: DEBUG)
        max_bytes: Max size before rotation (default: 50MB)
        backup_count: Number of backup files to keep (default: 5)
        fmt: Log message format
        clear_handlers: If True, clear existing handlers before adding new one

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if clear_handlers:
        logger.handlers.clear()

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    os.makedirs(log_dir, exist_ok=True)

    handler = _NFSSafeRotatingFileHandler(
        os.path.join(log_dir, log_file),
        mode='a',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def initialize_logger(log_dir, log_filename, process_type=None, use_custom_log_file=False):
    """Initialize all loggers for a process.

    Creates:
        main_logger: Primary application log
        cost_logger: API cost tracking
        time_memory_logger: Profiling data (main process only)

    Args:
        log_dir: Directory for log files
        log_filename: Filename for main log (used when use_custom_log_file=True or for Main process)
        process_type: One of "Main", "Sampler", "Evaluator", "Throughput"
        use_custom_log_file: If True, use log_filename instead of default per-type file
    """
    # Determine log file for main_logger based on process type
    if use_custom_log_file:
        main_log_file = log_filename
    elif process_type == "Sampler":
        main_log_file = 'samplers.log'
    elif process_type == "Evaluator":
        main_log_file = 'evaluators.log'
    else:
        main_log_file = log_filename

    # Clear handlers for child processes (they inherit parent's handlers after spawn)
    clear = process_type in ("Sampler", "Evaluator", "Throughput")

    # Main logger
    logger = create_logger('main_logger', log_dir, main_log_file, clear_handlers=clear)

    # Cost logger (API costs)
    create_logger(
        'cost_logger', log_dir, 'costs.log',
        level=logging.INFO,
        max_bytes=10*1024*1024,
        backup_count=3,
        fmt='%(asctime)s - %(message)s'
    )

    # Time/memory logger (profiling, main process only)
    if process_type == "Main":
        create_logger('time_memory_logger', log_dir, 'time_memory.log')

    return logger


def load_initial_programs(initial_functions_dir, strip_tags=False, logger=None):
    """Load initial functions from a directory.

    Args:
        initial_functions_dir: Path to directory containing .txt files with initial functions
        strip_tags: If True, remove description/code tags from function bodies.
            Default False lets evaluator's parse_llm_output handle tags consistently,
            preserving descriptions for prompt building.
        logger: Optional logger for status messages

    Returns:
        List of JSON strings ready to be published to evaluator queue
    """
    init_dir = Path(initial_functions_dir)
    if not init_dir.exists() or not init_dir.is_dir():
        if logger:
            logger.warning(f"Initial functions directory not found: {init_dir}")
        return []

    programs = []
    for txt_file in sorted(init_dir.glob("*.txt")):
        try:
            function_body = txt_file.read_text().strip()

            if strip_tags:
                function_body = re.sub(r'<description>.*?</description>\s*', '', function_body, flags=re.DOTALL)
                function_body = re.sub(r'<code>(.*?)</code>', r'\1', function_body, flags=re.DOTALL)
                function_body = function_body.strip()

            programs.append(json.dumps({
                "sample": function_body,
                "island_id": None,
                "version_generated": None,
                "expected_version": 0
            }))
            if logger:
                logger.info(f"Loaded initial function: {txt_file.name}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load {txt_file}: {e}")

    if logger:
        if programs:
            logger.info(f"Loaded {len(programs)} initial function(s)")
        else:
            logger.warning(f"No initial functions found in {init_dir}")

    return programs


def get_gpu_count():
    """Get number of GPUs available to this process.

    Respects CUDA_VISIBLE_DEVICES if set (e.g. by SLURM).
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([g for g in visible.split(",") if g.strip()])
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([l for l in result.stdout.split("\n") if l.startswith("GPU")])
    except Exception:
        return 0


def sampler_process_entry(config_path, device, log_dir, log_filename, sampler_id=0, use_parent_log=False):
    """Standalone sampler process entry point (spawn-compatible)."""
    # Load config first to determine GPU assignment
    config = load_config(config_path)
    tp_size = int(config.sampler.tensor_parallel_size)

    # Set CUDA_VISIBLE_DEVICES BEFORE importing anything that touches CUDA/PyTorch
    if config.sampler.use_local_vllm:
        # Pick this sampler's GPU(s) from the available pool.
        # Read CUDA_VISIBLE_DEVICES first so we respect SLURM GPU assignments.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            all_gpus = [g.strip() for g in visible.split(",") if g.strip()]
        else:
            all_gpus = [str(i) for i in range(get_gpu_count())]

        start = sampler_id * tp_size
        if start + tp_size > len(all_gpus):
            raise RuntimeError(
                f"Sampler {sampler_id} needs GPUs [{start}:{start + tp_size}], "
                f"but only {len(all_gpus)} available: {all_gpus}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(all_gpus[start:start + tp_size])
        print(f"Sampler {sampler_id}: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        device = "cuda:0"
    elif device is not None:
        # Explicit device override (e.g. from resource manager)
        device_id = device.split(":")[-1] if isinstance(device, str) and ":" in device else str(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"Sampler {sampler_id}: CUDA_VISIBLE_DEVICES={device_id}")
        device = "cuda:0"

    if hasattr(config.sampler, 'cache_dir') and config.sampler.cache_dir:
        os.environ["HF_HOME"] = config.sampler.cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(config.sampler.cache_dir, "hub")
        print(f"Sampler process: Set HF_HOME to {config.sampler.cache_dir}")

    from disfun import sampler
    from disfun.utils import rabbitmq

    logger = initialize_logger(log_dir, log_filename, process_type="Sampler", use_custom_log_file=use_parent_log)

    local_id = current_process().pid
    state = ProcessState()
    connection = None
    channel = None

    async def shutdown():
        if state.shutdown_started:
            return
        state.shutdown_started = True
        logger.info(f"Sampler {local_id}: Shutting down...")

        if state.task and not state.task.done():
            state.task.cancel()
            try:
                await asyncio.wait_for(state.task, timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                pass

        if state.instance:
            await state.instance.shutdown()
        else:
            # Fallback if instance never initialized
            if channel:
                try:
                    await channel.close()
                except Exception:
                    pass
            if connection:
                try:
                    await connection.close()
                except Exception:
                    pass

        loop.stop()

    async def run():
        nonlocal connection, channel
        try:
            # Initialize model BEFORE connecting to RabbitMQ (avoids heartbeat timeouts)
            logger.info(f"Sampler {local_id}: Initializing model {config.sampler.model} on device {device}...")

            sampler_seed = None
            if config.random_seed is not None:
                sampler_seed = hash((config.random_seed, sampler_id)) % (2**31)
                logger.info(f"Sampler {local_id}: Using seed {sampler_seed}")

            try:
                state.instance = sampler.Sampler(
                    config.sampler, rabbitmq_config=config, device=device, log_dir=log_dir,
                    random_seed=sampler_seed, sampler_id=sampler_id,
                    debug_samples=getattr(config.evaluator, 'debug_samples', False),
                )
                logger.info(f"Sampler {local_id}: Model loaded successfully.")
            except Exception as e:
                logger.error(f"Sampler {local_id}: Could not initialize model: {e}", exc_info=True)
                return

            # Connect to RabbitMQ after model is loaded
            logger.info(f"Sampler {local_id}: Connecting to RabbitMQ...")
            connection = await rabbitmq.create_connection(config)
            channel = await connection.channel()
            sampler_queue = await rabbitmq.declare_queue(channel, "sampler_queue")
            evaluator_queue = await rabbitmq.declare_queue(channel, "evaluator_queue")
            logger.info(f"Sampler {local_id}: Connected to RabbitMQ.")

            state.instance.connection = connection
            state.instance.channel = channel
            state.instance.sampler_queue = sampler_queue
            state.instance.evaluator_queue = evaluator_queue

            state.task = asyncio.create_task(state.instance.consume_and_process())
            await state.task

        except asyncio.CancelledError:
            logger.info(f"Sampler {local_id}: Cancelled.")
        except Exception as e:
            logger.error(f"Sampler {local_id}: Error: {e}")
        finally:
            if not state.shutdown_started:
                await shutdown()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handlers — track which signal killed us for exit code
    shutdown_task = None
    received_signal = [None]

    def on_signal(signum):
        nonlocal shutdown_task
        received_signal[0] = signum
        ppid = os.getppid()
        logger.warning(
            f"Sampler {local_id} (PID {os.getpid()}): Received {signal.Signals(signum).name} "
            f"(parent PID={ppid})"
        )
        if shutdown_task is None:
            shutdown_task = asyncio.create_task(shutdown())

    loop.add_signal_handler(signal.SIGTERM, lambda: on_signal(signal.SIGTERM))
    loop.add_signal_handler(signal.SIGINT, lambda: on_signal(signal.SIGINT))

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
        if received_signal[0] is not None:
            sys.exit(128 + received_signal[0])
        sys.exit(0)


def evaluator_process_entry(config_path, template, inputs, target_signatures, log_dir, log_filename, use_parent_log=False):
    """Standalone evaluator process entry point (spawn-compatible)."""
    from disfun.utils import rabbitmq
    import disfun.evaluator as evaluator_module

    config = load_config(config_path)
    logger = initialize_logger(log_dir, log_filename, process_type="Evaluator", use_custom_log_file=use_parent_log)

    local_id = current_process().pid
    state = ProcessState()

    connection_manager = rabbitmq.ConnectionManager(
        config=config,
        component_name=f"Evaluator {local_id}",
        queue_names=["evaluator_queue", "database_queue"],
        logger=logger
    )

    async def shutdown():
        if state.shutdown_started:
            return
        state.shutdown_started = True
        logger.info(f"Evaluator {local_id}: Shutting down...")

        if state.task and not state.task.done():
            state.task.cancel()
            try:
                await asyncio.wait_for(state.task, timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                pass

        if state.instance:
            await state.instance.shutdown()

        loop.stop()

    async def run():
        try:
            state.instance = evaluator_module.Evaluator(
                template=template,
                inputs=inputs,
                local_id=local_id,
                evaluator_config=config.evaluator,
                connection_manager=connection_manager,
                target_signatures=target_signatures,
                log_dir=log_dir,
            )

            state.task = asyncio.create_task(state.instance.consume_and_process())
            await state.task

        except asyncio.CancelledError:
            logger.info(f"Evaluator {local_id}: Cancelled.")
        except Exception as e:
            logger.error(f"Evaluator {local_id}: Error: {e}")
        finally:
            if not state.shutdown_started:
                await shutdown()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handlers — track which signal killed us for exit code
    shutdown_task = None
    received_signal = [None]  # mutable container for nested access

    signal_count = [0]  # Track how many times signal handler fires

    def on_signal(signum):
        nonlocal shutdown_task
        signal_count[0] += 1
        received_signal[0] = signum
        pid = os.getpid()
        ppid = os.getppid()

        # Only log details on first signal (avoid log spam from repeated deliveries)
        if signal_count[0] == 1:
            import traceback
            pgid = os.getpgrp()
            logger.warning(
                f"Evaluator {local_id} (PID {pid}, PGID {pgid}): Received {signal.Signals(signum).name} "
                f"(parent PID={ppid}) [first signal]"
            )
            logger.warning(f"Evaluator {local_id}: Signal stack trace:\n{''.join(traceback.format_stack())}")
        elif signal_count[0] == 2:
            logger.warning(
                f"Evaluator {local_id} (PID {pid}): Received {signal.Signals(signum).name} "
                f"again (suppressing further logs, count={signal_count[0]})"
            )

        if shutdown_task is None:
            shutdown_task = asyncio.create_task(shutdown())

    loop.add_signal_handler(signal.SIGTERM, lambda: on_signal(signal.SIGTERM))
    loop.add_signal_handler(signal.SIGINT, lambda: on_signal(signal.SIGINT))

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
        # Use distinct exit codes so health monitor can identify death cause:
        #   0   = normal exit (consume loop finished)
        #   130 = SIGINT  (128 + 2)
        #   143 = SIGTERM (128 + 15)
        if received_signal[0] is not None:
            sys.exit(128 + received_signal[0])
        sys.exit(0)
