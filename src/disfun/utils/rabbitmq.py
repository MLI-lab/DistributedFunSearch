"""RabbitMQ connection and queue management for DistributedFunSearch.

This module provides centralized RabbitMQ management for all components.
The RabbitMQManager handles connections, queue operations, and metrics tracking.

Usage:
    # Create central manager (typically in __main__.py)
    manager = RabbitMQManager(config)

    # Get connection for a component
    conn = await manager.get_connection("ProgramsDatabase", ["database_queue", "sampler_queue"])

    # Get queue metrics
    metrics = await manager.get_metrics()

    # Cleanup
    await manager.close_all()
"""

import logging
import asyncio
import aiohttp
import aio_pika
from yarl import URL


logger = logging.getLogger('main_logger')


# Queue names used in the system
QUEUE_NAMES = ["sampler_queue", "evaluator_queue", "database_queue"]


async def create_connection(config, timeout=60, heartbeat=0):
    """Create a robust RabbitMQ connection.

    Args:
        config: Configuration object with rabbitmq settings
        timeout: Connection timeout in seconds (default 60)
        heartbeat: Heartbeat interval in seconds (default 0 = disabled)

    Returns:
        aio_pika.Connection: Robust connection to RabbitMQ
    """
    effective_heartbeat = getattr(config.rabbitmq, 'heartbeat', heartbeat)
    effective_timeout = getattr(config.rabbitmq, 'connection_timeout', timeout)

    try:
        amqp_url = URL(
            f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@'
            f'{config.rabbitmq.host}:{config.rabbitmq.port}/{config.rabbitmq.vhost}'
        ).update_query(heartbeat=effective_heartbeat)
        return await aio_pika.connect_robust(amqp_url, timeout=effective_timeout)
    except Exception as e:
        logger.warning(f"RabbitMQ connection with vhost '{config.rabbitmq.vhost}' failed: {e}. Retrying without vhost...")
        amqp_url = URL(
            f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@'
            f'{config.rabbitmq.host}:{config.rabbitmq.port}/'
        ).update_query(heartbeat=effective_heartbeat)
        return await aio_pika.connect_robust(amqp_url, timeout=effective_timeout)


async def declare_queue(channel, queue_name: str):
    """Declare a queue with standard DistributedFunSearch settings."""
    return await channel.declare_queue(
        queue_name,
        durable=False,
        auto_delete=False,
        arguments={'x-consumer-timeout': 360000000}
    )


async def get_queue_stats(config, queue_name: str) -> tuple[int, int]:
    """Get message and consumer counts for a queue via Management API.

    Args:
        config: Configuration object with rabbitmq settings
        queue_name: Name of the queue to query

    Returns:
        Tuple of (message_count, consumer_count)
    """
    try:
        cfg = config.rabbitmq
        management_port = getattr(cfg, 'management_port', 15672)
        vhost = '%2F' if not cfg.vhost else cfg.vhost
        url = f"http://{cfg.host}:{management_port}/api/queues/{vhost}/{queue_name}"
        timeout = aiohttp.ClientTimeout(total=5)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, auth=aiohttp.BasicAuth(cfg.username, cfg.password)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('messages', 0), data.get('consumers', 0)
                return 0, 0
    except Exception:
        return 0, 0


class ConnectionManager:
    """Manages a single RabbitMQ connection for a component.

    Each component (Sampler, Evaluator, ProgramsDatabase) uses one ConnectionManager.
    Provides automatic reconnection with exponential backoff.
    """

    def __init__(self, config, component_name: str, queue_names: list[str],
                 logger: logging.Logger = None, timeout: int = 300):
        self.config = config
        self.component_name = component_name
        self.queue_names = queue_names
        self.logger = logger or logging.getLogger('main_logger')
        self.timeout = timeout

        self.connection = None
        self.channel = None
        self.queues = {}
        self._shutdown_requested = False

    def needs_reconnect(self) -> bool:
        """Check if connection needs to be (re)established."""
        return (
            self.connection is None or
            self.connection.is_closed or
            self.channel is None or
            self.channel.is_closed
        )

    async def ensure_connection(self) -> bool:
        """Ensure connection is alive, reconnecting if necessary."""
        if self.config is None:
            self.logger.warning(f"{self.component_name}: No RabbitMQ config")
            return False

        if not self.needs_reconnect():
            return True

        self.logger.info(f"{self.component_name}: Connecting to RabbitMQ...")
        try:
            await self.close(shutdown=False)
            self.connection = await create_connection(self.config, timeout=self.timeout)
            self.channel = await self.connection.channel()

            self.queues.clear()
            for queue_name in self.queue_names:
                self.queues[queue_name] = await declare_queue(self.channel, queue_name)

            self.logger.info(f"{self.component_name}: Connected")
            return True

        except Exception as e:
            self.logger.error(f"{self.component_name}: Connection failed: {e}")
            return False

    def request_shutdown(self):
        """Signal that shutdown is requested, stops retry loops."""
        self._shutdown_requested = True

    async def connect_with_retry(self) -> bool:
        """Connect with exponential backoff retry loop.

        Retries indefinitely until connected or shutdown requested.
        Handles all connection errors internally.

        Returns:
            True if connected, False if shutdown was requested.
        """
        if self.config is None:
            self.logger.warning(f"{self.component_name}: No config, cannot connect")
            return False

        rmq = self.config.rabbitmq
        delay = rmq.reconnect_delay
        max_delay = rmq.max_reconnect_delay

        while not self._shutdown_requested:
            try:
                if await self.ensure_connection():
                    return True

                # Connection failed, retry with backoff
                self.logger.warning(
                    f"{self.component_name}: Connection failed, retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, max_delay)

            except asyncio.CancelledError:
                self.logger.info(f"{self.component_name}: Connection cancelled")
                raise

            except Exception as e:
                self.logger.warning(
                    f"{self.component_name}: {type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                )
                await self.close(shutdown=False)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, max_delay)

        self.logger.info(f"{self.component_name}: Shutdown requested, stopping connection attempts")
        return False

    async def close(self, shutdown=True):
        """Close connection and channel.

        Args:
            shutdown: If True (default), also sets shutdown flag to prevent reconnection.
                      Use shutdown=False when closing for reconnection.
        """
        if shutdown:
            self.request_shutdown()
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
        except Exception as e:
            self.logger.debug(f"{self.component_name}: Error closing channel: {e}")

        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
        except Exception as e:
            self.logger.debug(f"{self.component_name}: Error closing connection: {e}")

        self.connection = None
        self.channel = None
        self.queues.clear()

    def get_queue(self, queue_name: str):
        """Get a declared queue by name."""
        return self.queues.get(queue_name)


class RabbitMQManager:
    """Central manager for all RabbitMQ operations.

    Manages connections for all components, tracks queue metrics, and provides
    unified access to RabbitMQ state.

    Attributes:
        config: Full application config with rabbitmq settings
        connections: Dict mapping component_name -> ConnectionManager
    """

    def __init__(self, config, logger: logging.Logger = None):
        """Initialize the manager.

        Args:
            config: Full config object with config.rabbitmq settings
            logger: Logger instance (uses main_logger if not provided)
        """
        self.config = config
        self.logger = logger or logging.getLogger('main_logger')
        self.connections: dict[str, ConnectionManager] = {}
        self._metrics_cache = {}
        self._metrics_cache_time = 0
        self._cache_ttl = 1.0  # Cache metrics for 1 second

    def get_connection(self, component_name: str, queue_names: list[str],
                       timeout: int = 300) -> ConnectionManager:
        """Get or create a ConnectionManager for a component.

        Args:
            component_name: Name of the component (e.g., "ProgramsDatabase")
            queue_names: List of queue names the component needs
            timeout: Connection timeout in seconds

        Returns:
            ConnectionManager for the component
        """
        if component_name not in self.connections:
            self.connections[component_name] = ConnectionManager(
                config=self.config,
                component_name=component_name,
                queue_names=queue_names,
                logger=self.logger,
                timeout=timeout
            )
        return self.connections[component_name]

    async def get_queue_depths(self) -> dict[str, int]:
        """Get message counts for all queues via Management API.

        Returns:
            Dict mapping queue_name -> message_count
        """
        depths = {}
        for queue_name in QUEUE_NAMES:
            messages, _ = await get_queue_stats(self.config, queue_name)
            depths[queue_name] = messages
        return depths

    async def get_queue_consumers(self) -> dict[str, int]:
        """Get consumer counts for all queues via Management API.

        Returns:
            Dict mapping queue_name -> consumer_count
        """
        consumers = {}
        for queue_name in QUEUE_NAMES:
            _, consumer_count = await get_queue_stats(self.config, queue_name)
            consumers[queue_name] = consumer_count
        return consumers

    async def get_metrics(self) -> dict:
        """Get all RabbitMQ metrics.

        Returns dict with:
            queue_depths: {queue_name: message_count}
            queue_consumers: {queue_name: consumer_count}
            connections: {component_name: is_connected}
            total_messages: sum of all queue depths
        """
        # Use cached metrics if fresh enough
        now = asyncio.get_event_loop().time()
        if now - self._metrics_cache_time < self._cache_ttl and self._metrics_cache:
            return self._metrics_cache

        depths = await self.get_queue_depths()
        consumers = await self.get_queue_consumers()

        connection_status = {}
        for name, conn in self.connections.items():
            connection_status[name] = not conn.needs_reconnect()

        metrics = {
            "queue_depths": depths,
            "queue_consumers": consumers,
            "connections": connection_status,
            "total_messages": sum(depths.values()),
            "sampler_queue_depth": depths.get("sampler_queue", 0),
            "evaluator_queue_depth": depths.get("evaluator_queue", 0),
            "database_queue_depth": depths.get("database_queue", 0),
        }

        self._metrics_cache = metrics
        self._metrics_cache_time = now
        return metrics

    async def close_all(self):
        """Close all managed connections."""
        for name, conn in self.connections.items():
            try:
                await conn.close()
                self.logger.debug(f"Closed connection for {name}")
            except Exception as e:
                self.logger.warning(f"Error closing {name} connection: {e}")
        self.connections.clear()