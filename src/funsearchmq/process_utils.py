"""
Utility functions and classes for process and connection management.

This module provides reusable patterns to reduce code duplication across
the DeCoSearch codebase.
"""

import asyncio
import signal
import logging
import torch.multiprocessing as mp
from yarl import URL
import aio_pika
from typing import Optional, Callable


async def create_rabbitmq_connection(config, timeout=300, heartbeat=172800):
    """
    Create a robust RabbitMQ connection with standard configuration.

    Args:
        config: Configuration object with rabbitmq settings
        timeout: Connection timeout in seconds
        heartbeat: Heartbeat interval in seconds (default 172800 = 2 days)
                   Set to 2 days for testing long-running experiments.
                   If connection errors occur before this timeout, they are
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
    Declare a queue with standard DeCoSearch settings.

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


class ManagedProcess:
    """
    Context manager for multiprocessing.Process that ensures proper cleanup.
    
    Usage:
        with ManagedProcess(target=my_func, args=(arg1,)) as proc:
            # Process is started automatically
            pass
        # Process is terminated and cleaned up automatically
    """
    
    def __init__(self, target, args=(), kwargs=None, name=None, timeout=10):
        """
        Initialize a managed process.
        
        Args:
            target: Function to run in the process
            args: Positional arguments for target
            kwargs: Keyword arguments for target
            name: Name for the process
            timeout: Timeout for graceful termination (seconds)
        """
        self.proc = mp.Process(
            target=target,
            args=args,
            kwargs=kwargs or {},
            name=name
        )
        self.timeout = timeout
        self.logger = logging.getLogger('main_logger')
    
    def __enter__(self):
        """Start the process and return it."""
        self.proc.start()
        if self.logger:
            self.logger.debug(f"Started process {self.proc.name} (PID: {self.proc.pid})")
        return self.proc
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Terminate the process gracefully."""
        if self.proc.is_alive():
            self.proc.terminate()
            if self.logger:
                self.logger.debug(f"Sent SIGTERM to {self.proc.name} (PID: {self.proc.pid})")
            
            self.proc.join(timeout=self.timeout)
            
            if self.proc.is_alive():
                if self.logger:
                    self.logger.warning(
                        f"Process {self.proc.name} (PID: {self.proc.pid}) did not terminate "
                        f"in {self.timeout}s. Sending SIGKILL."
                    )
                self.proc.kill()
                self.proc.join()
        
        if self.logger:
            self.logger.debug(f"Process {self.proc.name} (PID: {self.proc.pid}) terminated")
        
        return False  # Don't suppress exceptions


class ConnectionManager:
    """
    Context manager for RabbitMQ connections and channels.
    
    Usage:
        async with ConnectionManager(config) as (connection, channel):
            # Use connection and channel
            pass
        # Connection and channel are closed automatically
    """
    
    def __init__(self, config, timeout=300):
        """
        Initialize connection manager.
        
        Args:
            config: Configuration object with rabbitmq settings
            timeout: Connection timeout in seconds
        """
        self.config = config
        self.timeout = timeout
        self.connection = None
        self.channel = None
    
    async def __aenter__(self):
        """Create connection and channel."""
        self.connection = await create_rabbitmq_connection(self.config, self.timeout)
        self.channel = await self.connection.channel()
        return self.connection, self.channel
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close channel and connection."""
        if self.channel:
            try:
                await self.channel.close()
            except Exception:
                pass
        
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                pass

        return False  # Don't suppress exceptions


async def with_reconnection(consume_func: Callable, logger: logging.Logger,
                           component_name: str = "Component",
                           initial_delay: float = 5.0,
                           max_delay: float = 60.0):
    """
    Wrapper that adds automatic reconnection logic to consume functions.

    When a connection error occurs, this wrapper will:
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
            # If consume_func exits normally, break the loop
            break

        except asyncio.CancelledError:
            # Shutdown requested - exit reconnection loop
            logger.info(f"{component_name} shutting down, exiting reconnection loop.")
            break

        except Exception as e:
            # Connection error occurred - log details and retry
            logger.error(
                f"{component_name} connection error: {e}\n"
                f"This can occur due to:\n"
                f"  - RabbitMQ connection reset (network issues, heartbeat timeout)\n"
                f"  - RabbitMQ server overload with many simultaneous connections\n"
                f"  - Network interruptions in cluster environment\n"
                f"Attempting to reconnect in {reconnect_delay:.1f} seconds..."
            )

            await asyncio.sleep(reconnect_delay)

            # Exponential backoff up to max
            reconnect_delay = min(reconnect_delay * 1.5, max_delay)

            logger.info(f"{component_name} reconnecting after {type(e).__name__}...")
            continue  # Retry connection
