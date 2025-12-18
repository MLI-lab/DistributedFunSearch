"""
Utility functions and classes for process and connection management.

This module provides reusable patterns to reduce code duplication across
the DistributedFunSearch codebase.
"""

import os
import asyncio
import signal
import logging
import importlib.util
from logging.handlers import RotatingFileHandler
from yarl import URL
import aio_pika
from collections.abc import Callable


def convert_tuple_keys(obj):
    """
    Recursively convert tuple keys to strings for JSON serialization.

    Useful when passing dataclass configs to wandb.init(), since JSON
    doesn't support tuple keys (e.g., {(7, 2): 5} -> {"(7, 2)": 5}).

    Args:
        obj: Dict, list, or other object to process

    Returns:
        Object with tuple keys converted to strings
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: convert_tuple_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuple_keys(item) for item in obj]
    return obj


def load_config(config_path):
    """
    Dynamically load a configuration from a given file path.

    Supports:
        - .py files: Dynamically loaded as Python modules (must define Config class)
        - .pkl files: Loaded via cloudpickle (for sweep configs with modified params)

    Args:
        config_path: Path to the configuration file (.py or .pkl)

    Returns:
        Config object instance
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Handle pickle files (used by throughput_sweep.py)
    if config_path.endswith('.pkl'):
        import cloudpickle
        with open(config_path, 'rb') as f:
            return cloudpickle.load(f)

    # Handle Python files (original behavior)
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "Config"):
        raise ValueError(f"The configuration file at {config_path} must define a 'Config' class.")

    return config_module.Config()


def initialize_logger(log_dir, log_filename, process_type=None, use_custom_log_file=False):
    """
    Initialize logger for process (works for both parent and child processes).

    Args:
        log_dir: Directory containing the log file
        log_filename: Name of the log file to write to
        process_type: Type of process ("Sampler", "Evaluator", "Throughput", or None for main)
        use_custom_log_file: If True, use log_filename even for Sampler/Evaluator process types

    Returns:
        Logger instance
    """
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)

    # For child processes or new throughput runs - clear any inherited handlers
    if process_type in ("Sampler", "Evaluator", "Throughput"):
        logger.handlers.clear()

    # Only add handler if logger doesn't have any yet
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # If use_custom_log_file is True, always use the provided log_filename
        if use_custom_log_file:
            log_file_path = os.path.join(log_dir, log_filename)
            handler = RotatingFileHandler(log_file_path, mode='a', maxBytes=50*1024*1024, backupCount=5)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Samplers: all log to shared samplers.log file
        # Use regular FileHandler to avoid rotation race conditions with multiple processes
        elif process_type == "Sampler":
            sampler_handler = logging.FileHandler(
                os.path.join(log_dir, 'samplers.log'),
                mode='a',
            )
            sampler_handler.setFormatter(formatter)
            logger.addHandler(sampler_handler)

        # Evaluators: all log to shared evaluators.log file
        # Use regular FileHandler to avoid rotation race conditions with multiple processes
        elif process_type == "Evaluator":
            evaluator_handler = logging.FileHandler(
                os.path.join(log_dir, 'evaluators.log'),
                mode='a',
            )
            evaluator_handler.setFormatter(formatter)
            logger.addHandler(evaluator_handler)

        # Main process/database: log to main.log
        else:
            log_file_path = os.path.join(log_dir, log_filename)
            handler = RotatingFileHandler(log_file_path, mode='a', maxBytes=50*1024*1024, backupCount=5)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.propagate = False

    # Setup cost logger (shared file across all processes)
    cost_logger = logging.getLogger('cost_logger')
    if not cost_logger.handlers:
        # Use RotatingFileHandler: rotates when file reaches 10 MB - keeps 3 backups
        cost_handler = RotatingFileHandler(os.path.join(log_dir, 'costs.log'), mode='a', maxBytes=10*1024*1024, backupCount=3)
        cost_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        cost_logger.addHandler(cost_handler)
        cost_logger.setLevel(logging.INFO)
        cost_logger.propagate = False

    return logger


async def create_rabbitmq_connection(config, timeout=60, heartbeat=0):
    """
    Create a robust RabbitMQ connection with standard configuration.

    Args:
        config: Configuration object with rabbitmq settings
        timeout: Connection timeout in seconds (default 60)
        heartbeat: Heartbeat interval in seconds (default 0 = disabled)
                   Disabled by default to prevent false disconnects when
                   RabbitMQ Erlang VM is CPU-starved under heavy load.

    Returns:
        aio_pika.Connection: Robust connection to RabbitMQ
    """
    # Use config heartbeat if available, otherwise use default
    effective_heartbeat = getattr(config.rabbitmq, 'heartbeat', heartbeat)
    effective_timeout = getattr(config.rabbitmq, 'connection_timeout', timeout)

    logger = logging.getLogger('main_logger')

    try:
        amqp_url = URL(
            f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@'
            f'{config.rabbitmq.host}:{config.rabbitmq.port}/{config.rabbitmq.vhost}'
        ).update_query(heartbeat=effective_heartbeat)
        return await aio_pika.connect_robust(amqp_url, timeout=effective_timeout)
    except Exception as e:
        # Log the error before retrying without vhost
        logger.warning(f"RabbitMQ connection with vhost '{config.rabbitmq.vhost}' failed: {e}. Retrying without vhost...")
        amqp_url = URL(
            f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@'
            f'{config.rabbitmq.host}:{config.rabbitmq.port}/'
        ).update_query(heartbeat=effective_heartbeat)
        return await aio_pika.connect_robust(amqp_url, timeout=effective_timeout)


async def declare_standard_queue(channel, queue_name: str):
    """
    Declare a queue with standard DistributedFunSearch settings.

    Args:
        channel: aio_pika channel
        queue_name: Name of the queue to declare

    Returns:
        aio_pika.Queue: Declared queue
    """
    return await channel.declare_queue(
        queue_name,
        durable=False,
        auto_delete=False,  # Changed to False to prevent queue deletion when consumers disconnect
        arguments={'x-consumer-timeout': 360000000}
    )


def setup_signal_handlers(loop, process_type: str, local_id: int, logger: logging.Logger,
                          graceful_shutdown_func: Callable):
    """
    Set up standard SIGTERM and SIGINT handlers for a process.

    Args:
        loop: asyncio event loop
        process_type: Type of process (e.g., "Sampler", "Evaluator")
        local_id: Process ID or identifier
        logger: Logger instance
        graceful_shutdown_func: Async function to call for graceful shutdown
    """
    shutdown_task = None

    def shutdown_callback():
        nonlocal shutdown_task
        if shutdown_task is None:
            logger.info(f"{process_type} {local_id}: Received shutdown signal, scheduling graceful shutdown.")
            shutdown_task = asyncio.create_task(graceful_shutdown_func())
        else:
            logger.debug(f"{process_type} {local_id}: Shutdown already in progress, ignoring duplicate signal.")

    loop.add_signal_handler(signal.SIGTERM, shutdown_callback)
    loop.add_signal_handler(signal.SIGINT, shutdown_callback)


async def graceful_shutdown(component_type: str, local_id: int, logger: logging.Logger,
                           loop, connection, channel, task, instance,
                           cleanup_done_flag: dict):
    """
    Shared graceful shutdown logic for all process types.

    Args:
        component_type: Type of component ("Sampler" or "Evaluator")
        local_id: Process ID or identifier
        logger: Logger instance
        loop: Asyncio event loop
        connection: RabbitMQ connection (may be stale if instance reconnected)
        channel: RabbitMQ channel (may be stale if instance reconnected)
        task: Async task to cancel
        instance: Component instance to cleanup (Sampler or Evaluator)
        cleanup_done_flag: Dict with 'done' key to track cleanup state
    """
    if cleanup_done_flag.get('done', False):
        return

    logger.info(f"{component_type} {local_id}: Initiating graceful shutdown...")

    # CRITICAL: Set shutdown flag FIRST to stop reconnection attempts
    if instance and hasattr(instance, 'request_shutdown'):
        instance.request_shutdown()
        logger.info(f"{component_type} {local_id}: Shutdown flag set")

    # Cancel consume task - it will see the flag and exit cleanly
    if task and not task.done():
        logger.info(f"{component_type} {local_id}: Cancelling consume task...")
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass

    # Cleanup instance - prefer async_cleanup for Samplers (handles reconnected connections)
    if instance:
        try:
            logger.info(f"{component_type} {local_id}: Cleaning up {component_type.lower()} instance...")
            if hasattr(instance, 'async_cleanup'):
                # Sampler: use async_cleanup to close current (possibly reconnected) connections
                await instance.async_cleanup()
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()
            if hasattr(instance, 'cleanup'):
                instance.cleanup()
        except Exception as e:
            logger.error(f"{component_type} {local_id}: Error during cleanup: {e}")
    else:
        # Fallback: Only close original connections if instance never initialized
        # (otherwise instance.async_cleanup handles its own connections)
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

    cleanup_done_flag['done'] = True
    logger.info(f"{component_type} {local_id}: Graceful shutdown complete.")
    loop.stop()


class RabbitMQConnectionManager:
    """
    Shared RabbitMQ connection management for all components.

    Provides unified connection, reconnection, and cleanup logic to reduce
    code duplication across Sampler, Evaluator, and ProgramsDatabase.

    Usage:
        # In component __init__:
        self._conn_manager = RabbitMQConnectionManager(
            config=rabbitmq_config,
            component_name="Sampler",
            queue_names=["sampler_queue", "evaluator_queue"],
            logger=logger
        )

        # To ensure connection before operations:
        if await self._conn_manager.ensure_connection():
            # Use self._conn_manager.connection, channel, queues

        # For cleanup:
        await self._conn_manager.close()
    """

    def __init__(self, config, component_name: str, queue_names: list[str],
                 logger: logging.Logger = None, timeout: int = 300):
        """
        Initialize the connection manager.

        Args:
            config: Full config object with rabbitmq settings
            component_name: Name for logging (e.g., "Sampler", "Evaluator")
            queue_names: List of queue names to declare on connection
            logger: Logger instance (uses main_logger if not provided)
            timeout: Connection timeout in seconds
        """
        self.config = config
        self.component_name = component_name
        self.queue_names = queue_names
        self.logger = logger or logging.getLogger('main_logger')
        self.timeout = timeout

        # Connection state
        self.connection = None
        self.channel = None
        self.queues = {}  # Maps queue_name -> queue object

    def needs_reconnect(self) -> bool:
        """Check if connection needs to be (re)established."""
        return (
            self.connection is None or
            self.connection.is_closed or
            self.channel is None or
            self.channel.is_closed
        )

    async def ensure_connection(self) -> bool:
        """
        Ensure RabbitMQ connection is alive, reconnecting if necessary.

        Returns:
            bool: True if connection is valid, False if reconnection failed
        """
        if self.config is None:
            self.logger.warning(f"{self.component_name}: No RabbitMQ config, cannot connect")
            return False

        if not self.needs_reconnect():
            return True  # Connection is already valid

        self.logger.info(f"{self.component_name}: Establishing RabbitMQ connection...")
        try:
            # Close any stale connections first
            await self.close()

            # Create fresh connection
            self.connection = await create_rabbitmq_connection(self.config, timeout=self.timeout)
            self.channel = await self.connection.channel()

            # Declare all queues
            self.queues.clear()
            for queue_name in self.queue_names:
                self.queues[queue_name] = await declare_standard_queue(self.channel, queue_name)

            self.logger.info(f"{self.component_name}: RabbitMQ connection established successfully")
            return True

        except Exception as e:
            self.logger.error(f"{self.component_name}: Failed to connect: {e}")
            return False

    async def close(self):
        """Safely close existing RabbitMQ connection and channel."""
        try:
            if self.channel is not None and not self.channel.is_closed:
                await self.channel.close()
        except Exception as e:
            self.logger.debug(f"{self.component_name}: Error closing channel: {e}")

        try:
            if self.connection is not None and not self.connection.is_closed:
                await self.connection.close()
        except Exception as e:
            self.logger.debug(f"{self.component_name}: Error closing connection: {e}")

        self.connection = None
        self.channel = None
        self.queues.clear()

    def get_queue(self, queue_name: str):
        """Get a declared queue by name."""
        return self.queues.get(queue_name)
