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


async def create_rabbitmq_connection(config, timeout=60, heartbeat=60):
    """
    Create a robust RabbitMQ connection with standard configuration.

    Args:
        config: Configuration object with rabbitmq settings
        timeout: Connection timeout in seconds (default 60)
        heartbeat: Heartbeat interval in seconds (default 60)
                   RabbitMQ detects dead connections within 2*heartbeat (~2 min).
                   This enables faster failure detection in cluster scenarios.

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
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
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


async def with_reconnection(consume_func: Callable, logger: logging.Logger,
                           component_name: str = "Component",
                           initial_delay: float = 5.0,
                           max_delay: float = 60.0):
    """
    Wrapper that adds automatic reconnection logic to consume functions.

    Note: For new code, prefer using RabbitMQConsumerMixin._run_with_reconnection()
    which also handles shutdown coordination. This function is kept for ProgramsDatabase
    which runs in main process and doesn't need shutdown flag coordination.
    """
    reconnect_delay = initial_delay

    while True:
        try:
            await consume_func()
            break
        except asyncio.CancelledError:
            logger.info(f"{component_name} shutting down.")
            break
        except Exception as e:
            logger.error(f"{component_name} error: {e}. Reconnecting in {reconnect_delay:.1f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_delay)
