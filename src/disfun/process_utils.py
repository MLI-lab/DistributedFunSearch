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
import torch.multiprocessing as mp
from logging import FileHandler
from logging.handlers import RotatingFileHandler
from yarl import URL
import aio_pika
from typing import Optional, Callable


def load_config(config_path):
    """
    Dynamically load a configuration module from a given file path.

    Args:
        config_path: Path to the configuration file

    Returns:
        Config object instance
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

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
        process_type: Type of process ("Sampler", "Evaluator", or None for main)
        use_custom_log_file: If True, use log_filename even for Sampler/Evaluator process types

    Returns:
        Logger instance
    """
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)

    # For child processes (Sampler/Evaluator) - clear any inherited handlers from parent
    if process_type in ("Sampler", "Evaluator"):
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
        elif process_type == "Sampler":
            sampler_handler = RotatingFileHandler(
                os.path.join(log_dir, 'samplers.log'),
                mode='a',
                maxBytes=50*1024*1024,
                backupCount=5
            )
            sampler_handler.setFormatter(formatter)
            logger.addHandler(sampler_handler)

        # Evaluators: all log to shared evaluators.log file
        elif process_type == "Evaluator":
            evaluator_handler = RotatingFileHandler(
                os.path.join(log_dir, 'evaluators.log'),
                mode='a',
                maxBytes=50*1024*1024,
                backupCount=5
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


async def create_rabbitmq_connection(config, timeout=300, heartbeat=172800):
    """
    Create a robust RabbitMQ connection with standard configuration.

    Args:
        config: Configuration object with rabbitmq settings
        timeout: Connection timeout in seconds
        heartbeat: Heartbeat interval in seconds (default 172800 = 2 days)
                   Set to 2 days for testing long-running experiments.
                   If connection errors occur before this timeout - they are
                   likely due to network issues or RabbitMQ server resource limits,
                   not heartbeat timeouts.

    Returns:
        aio_pika.Connection: Robust connection to RabbitMQ
    """
    try:
        amqp_url = URL(
            f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@'
            f'{config.rabbitmq.host}:{config.rabbitmq.port}/{config.rabbitmq.vhost}'
        ).update_query(heartbeat=heartbeat)
        return await aio_pika.connect_robust(amqp_url, timeout=timeout)
    except Exception:
        # Try without vhost if it fails
        amqp_url = URL(
            f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@'
            f'{config.rabbitmq.host}:{config.rabbitmq.port}/'
        ).update_query(heartbeat=heartbeat)
        return await aio_pika.connect_robust(amqp_url, timeout=timeout)


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
        connection: RabbitMQ connection
        channel: RabbitMQ channel
        task: Async task to cancel
        instance: Component instance to cleanup (Sampler or Evaluator)
        cleanup_done_flag: Dict with 'done' key to track cleanup state
    """
    if cleanup_done_flag.get('done', False):
        return

    logger.info(f"{component_type} {local_id}: Initiating graceful shutdown...")

    # Cancel consume task
    if task and not task.done():
        logger.info(f"{component_type} {local_id}: Cancelling consume task...")
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    # Cleanup instance
    if instance:
        try:
            logger.info(f"{component_type} {local_id}: Cleaning up {component_type.lower()} instance...")
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()
            elif hasattr(instance, 'cleanup'):
                instance.cleanup()
        except Exception as e:
            logger.error(f"{component_type} {local_id}: Error during cleanup: {e}")

    # Close connections (simplified - no verbose error handling)
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


async def with_reconnection(consume_func: Callable, logger: logging.Logger,
                           component_name: str = "Component",
                           initial_delay: float = 5.0,
                           max_delay: float = 60.0):
    """
    Wrapper that adds automatic reconnection logic to consume functions.

    When a connection error occurs - this wrapper will:
    1. Log the error with helpful context
    2. Wait with exponential backoff
    3. Retry the consume function
    4. Exit cleanly on cancellation signals

    Args:
        consume_func: Async function to wrap (should contain the consume loop)
        logger: Logger instance for status messages
        component_name: Name for log messages (e.g., "Evaluator", "Sampler")
        initial_delay: Initial reconnection delay in seconds (default: 5)
        max_delay: Maximum reconnection delay in seconds (default: 60)

    Example:
        async def consume_loop():
            async with queue.iterator() as stream:
                async for message in stream:
                    await process(message)

        await with_reconnection(consume_loop, logger, "Evaluator")
    """
    reconnect_delay = initial_delay

    while True:  # Reconnection loop
        try:
            await consume_func()
            # If consume_func exits normally - break the loop
            break

        except asyncio.CancelledError:
            # Shutdown requested - exit reconnection loop
            logger.info(f"{component_name} shutting down, exiting reconnection loop.")
            break

        except Exception as e:
            # Connection error occurred - log details and retry
            logger.error(
                f"{component_name} connection error: {e}\n"
                f"Attempting to reconnect in {reconnect_delay:.1f} seconds..."
            )

            await asyncio.sleep(reconnect_delay)

            # Exponential backoff up to max
            reconnect_delay = min(reconnect_delay * 1.5, max_delay)

            logger.info(f"{component_name} reconnecting after {type(e).__name__}...")
            continue  # Retry connection
