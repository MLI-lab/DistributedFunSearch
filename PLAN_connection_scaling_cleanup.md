# Comprehensive Plan: Fix Scaling, Cleanup, and RabbitMQ Connection Logic

## Problem Summary

The current implementation has three interconnected issues:
1. **Cleanup logic** - Inconsistent between components, queue deletion race conditions
2. **Scaling logic** - Too slow to detect failures (4+ min delay), aggressive scale-down
3. **Connection logic** - 2-day heartbeat, evaluator has no reconnection, stale references

These interact badly: when network fails, connections hang for hours, scaling doesn't detect it quickly, and cleanup doesn't properly release resources.

---

## CRITICAL: How Systems Must Coordinate

### The Core Problem: Reconnection vs Shutdown Conflict

```
Current broken scenario (Ctrl+C):

1. User presses Ctrl+C
2. Main process sends SIGTERM to sampler
3. Sampler's signal handler triggers graceful_shutdown()
4. BUT: Sampler's consume_and_process() is in a while True reconnection loop
5. Signal handler cancels task, but exception handling catches CancelledError
6. Reconnection loop might try to reconnect WHILE being killed
7. Race condition → hang or zombie process
```

```
Current broken scenario (Scale-down):

1. ResourceManager decides to terminate a sampler (queue empty)
2. Calls terminate_process() → sends SIGTERM
3. Sampler receives SIGTERM
4. BUT: Sampler doesn't know this is intentional termination
5. Sampler's reconnection loop might try to reconnect
6. ResourceManager waits 30s... then force kills
```

### Solution: Shutdown Flag That Stops Reconnection

The key insight: **Reconnection should STOP when shutdown is requested**.

We use the existing `cleanup_done` dict as a shutdown flag:

```python
# In process_entry.py - already exists!
cleanup_done = {'done': False}  # This becomes our shutdown flag
```

The reconnection loop must check this flag:

```python
# In sampler.py consume_and_process()
async def consume_and_process(self):
    while True:  # Reconnection loop
        # CHECK SHUTDOWN FLAG FIRST
        if self._shutdown_requested:
            logger.info("Shutdown requested, exiting reconnection loop")
            break

        try:
            await self._ensure_connection()
            await self._consume_loop()
        except asyncio.CancelledError:
            # CancelledError = shutdown signal, exit immediately
            break
        except ConnectionError:
            if self._shutdown_requested:
                break  # Don't reconnect if shutting down
            await asyncio.sleep(reconnect_delay)
            continue
```

### Flow Diagram: How It Works Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NORMAL OPERATION                                     │
│                                                                              │
│  Sampler.consume_and_process()                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  while True:  # Reconnection loop                                    │   │
│  │      if self._shutdown_requested: break  ◄─── Check before reconnect│   │
│  │      try:                                                            │   │
│  │          await _ensure_connection()  # Reconnect if needed           │   │
│  │          await _consume_loop()       # Process messages              │   │
│  │      except CancelledError:                                          │   │
│  │          break  ◄─── Exit on cancellation                           │   │
│  │      except ConnectionError:                                         │   │
│  │          if self._shutdown_requested: break ◄─── Don't reconnect    │   │
│  │          await sleep(backoff)                                        │   │
│  │          continue  # Retry connection                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         KEYBOARD INTERRUPT (Ctrl+C)                          │
│                                                                              │
│  1. Main receives SIGINT                                                    │
│  2. Main sends SIGTERM to all child processes                               │
│  3. Sampler receives SIGTERM                                                │
│     └─► Signal handler calls graceful_shutdown()                            │
│         └─► Sets self._shutdown_requested = True  ◄─── FLAG SET             │
│         └─► Cancels consume task (raises CancelledError)                    │
│  4. consume_and_process catches CancelledError                              │
│     └─► Sees it's CancelledError → breaks immediately                       │
│     └─► Does NOT try to reconnect                                           │
│  5. Cleanup runs (close connections, GPU memory, etc.)                      │
│  6. Process exits cleanly                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCALE-DOWN                                           │
│                                                                              │
│  1. ResourceManager.terminate_process() called                              │
│  2. Sends SIGTERM to sampler                                                │
│  3. Sampler receives SIGTERM                                                │
│     └─► Same as Ctrl+C: graceful_shutdown() sets _shutdown_requested        │
│  4. Sampler exits without trying to reconnect                               │
│  5. ResourceManager sees process exited                                     │
│  6. Frees GPU device from process_to_device_map                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         NETWORK FAILURE (should reconnect)                   │
│                                                                              │
│  1. Network drops, RabbitMQ connection lost                                 │
│  2. consume_and_process catches ConnectionError                             │
│  3. Checks: self._shutdown_requested == False  ◄─── No shutdown, reconnect │
│  4. Waits with backoff, then continues loop                                 │
│  5. _ensure_connection() creates new connection                             │
│  6. _consume_loop() resumes processing                                      │
│  7. ResourceManager sees consumer_count > 0                                 │
│     └─► Does NOT spawn duplicate sampler                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scale-Down vs Network Failure: How to Tell the Difference

| Scenario | SIGTERM Received? | _shutdown_requested | Action |
|----------|-------------------|---------------------|--------|
| Network failure | No | False | Reconnect |
| Ctrl+C | Yes | True | Exit cleanly |
| Scale-down | Yes | True | Exit cleanly |
| RabbitMQ restart | No | False | Reconnect |

The difference is simple: **SIGTERM sets the shutdown flag**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Main Process                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ TaskManager  │  │ResourceManager│  │ _shutdown() cleanup         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                        │
         ▼                    ▼                        ▼
┌─────────────┐      ┌─────────────┐         ┌─────────────┐
│  Sampler    │      │  Evaluator  │         │  Database   │
│ (has _ensure│      │ (NO _ensure │         │ (NO reconnect)│
│ _connection)│      │ _connection)│         │             │
└─────────────┘      └─────────────┘         └─────────────┘
         │                    │                        │
         └────────────────────┴────────────────────────┘
                              │
                              ▼
                      ┌─────────────┐
                      │  RabbitMQ   │
                      │  (heartbeat │
                      │   = 2 days!)│
                      └─────────────┘
```

---

## Phase 0: Add Shutdown Flag Coordination (MUST DO FIRST)

This is the foundation that makes everything else work together.

### 0.1 Add `_shutdown_requested` Flag to Sampler

**File:** `src/disfun/sampler.py`

**Add to `__init__` (around line 520):**
```python
def __init__(self, connection, channel, ...):
    # ... existing code ...
    self._shutdown_requested = False  # NEW: Flag to stop reconnection on shutdown
```

**Add method to set it:**
```python
def request_shutdown(self):
    """Signal that shutdown is requested - stops reconnection attempts."""
    self._shutdown_requested = True
```

### 0.2 Update Sampler `consume_and_process()` to Check Flag

**File:** `src/disfun/sampler.py:679-732`

**Update the reconnection loop:**
```python
async def consume_and_process(self) -> None:
    """Main consume loop with automatic connection recovery."""
    reconnect_delay = self._reconnect_delay

    while True:
        # CHECK SHUTDOWN FLAG AT TOP OF LOOP
        if self._shutdown_requested:
            logger.info(f"Sampler ({self._config.model}): Shutdown requested, exiting consume loop")
            break

        try:
            # Ensure connection is alive (reconnect if needed)
            if self._rabbitmq_config is not None:
                connected = await self._ensure_connection()
                if not connected:
                    # Check shutdown flag before sleeping
                    if self._shutdown_requested:
                        break
                    logger.error(f"Sampler: Connection failed, retrying in {reconnect_delay:.1f}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, self._max_reconnect_delay)
                    continue

            reconnect_delay = self._reconnect_delay
            await _consume_loop()
            break

        except asyncio.CancelledError:
            # CancelledError means shutdown - exit immediately, don't reconnect
            logger.info(f"Sampler ({self._config.model}): Cancelled, exiting...")
            break

        except (aio_pika.exceptions.AMQPConnectionError, ...) as e:
            # CHECK SHUTDOWN FLAG BEFORE RECONNECTING
            if self._shutdown_requested:
                logger.info(f"Sampler: Connection error during shutdown, exiting")
                break

            logger.warning(f"Sampler: Connection error: {e}. Reconnecting...")
            await self._close_connection()
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, self._max_reconnect_delay)
            continue

        except Exception as e:
            if self._shutdown_requested:
                break
            # ... existing error handling ...
```

### 0.3 Update `graceful_shutdown()` to Set Flag

**File:** `src/disfun/process_utils.py:196-257`

**Update to set shutdown flag on instance:**
```python
async def graceful_shutdown(component_type: str, local_id: int, logger: logging.Logger,
                           loop, connection, channel, task, instance,
                           cleanup_done_flag: dict):
    if cleanup_done_flag.get('done', False):
        return

    logger.info(f"{component_type} {local_id}: Initiating graceful shutdown...")

    # SET SHUTDOWN FLAG ON INSTANCE FIRST (before cancelling task)
    if instance and hasattr(instance, 'request_shutdown'):
        instance.request_shutdown()
        logger.info(f"{component_type} {local_id}: Shutdown flag set on instance")

    # NOW cancel consume task - it will see the flag and exit cleanly
    if task and not task.done():
        logger.info(f"{component_type} {local_id}: Cancelling consume task...")
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5)  # Increased timeout
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    # ... rest of cleanup ...
```

### 0.4 Add Same Pattern to Evaluator

**File:** `src/disfun/evaluator.py`

**Add to `__init__`:**
```python
def __init__(self, ...):
    # ... existing code ...
    self._shutdown_requested = False

def request_shutdown(self):
    """Signal that shutdown is requested."""
    self._shutdown_requested = True
```

**Check in consume loop (when we add reconnection).**

### 0.5 Why This Works

```
Scenario: Ctrl+C pressed

BEFORE (broken):
1. SIGTERM received
2. graceful_shutdown() cancels task
3. CancelledError raised in consume_and_process()
4. Exception handler catches it... might try to reconnect
5. Hang/race condition

AFTER (fixed):
1. SIGTERM received
2. graceful_shutdown() sets _shutdown_requested = True FIRST
3. graceful_shutdown() cancels task
4. CancelledError raised in consume_and_process()
5. Exception handler sees CancelledError → breaks immediately
6. Even if some code path misses CancelledError, the flag check at
   top of while loop catches it on next iteration
7. Clean exit
```

---

## Phase 1: Fix RabbitMQ Connection Settings (CRITICAL - Do First)

### 1.1 Reduce Heartbeat Timeout

**File:** `src/disfun/process_utils.py:120`

**Current:**
```python
async def create_rabbitmq_connection(config, timeout=300, heartbeat=172800):
    # heartbeat=172800 = 2 days - dead connections hang forever
```

**Change to:**
```python
async def create_rabbitmq_connection(config, timeout=60, heartbeat=60):
    """
    Args:
        timeout: Connection timeout (reduced from 300s to 60s for faster failure detection)
        heartbeat: Heartbeat interval (reduced from 2 days to 60s)
                   RabbitMQ will detect dead connections within 2*heartbeat = 120s
    """
```

**Why:**
- Current 2-day heartbeat means RabbitMQ never detects dead connections
- 60s heartbeat = dead connection detected within ~2 minutes
- Matches typical cluster network timeout expectations

### 1.2 Add Connection Health Check Config

**File:** `src/experiments/experiment1/config.py` - Add to RabbitMQConfig

```python
@dataclasses.dataclass(frozen=True)
class RabbitMQConfig:
    # ... existing fields ...
    heartbeat: int = 60  # Heartbeat interval in seconds (default: 60)
    connection_timeout: int = 30  # Connection timeout in seconds (default: 30)
    prefetch_count: int = 10  # Messages to prefetch per consumer
```

### 1.3 Log Connection Errors (Don't Swallow)

**File:** `src/disfun/process_utils.py:136-148`

**Current:**
```python
try:
    return await aio_pika.connect_robust(amqp_url, timeout=timeout)
except Exception:  # Silent fallback!
    amqp_url = URL(...).update_query(heartbeat=heartbeat)
    return await aio_pika.connect_robust(amqp_url, timeout=timeout)
```

**Change to:**
```python
try:
    return await aio_pika.connect_robust(amqp_url, timeout=timeout)
except Exception as e:
    logger.warning(f"Connection with vhost failed: {e}. Retrying without vhost...")
    amqp_url = URL(...).update_query(heartbeat=heartbeat)
    return await aio_pika.connect_robust(amqp_url, timeout=timeout)
```

---

## Phase 2: Unify Reconnection Logic (HIGH PRIORITY)

### 2.1 Add `_ensure_connection()` to Evaluator

**File:** `src/disfun/evaluator.py`

The Sampler has a robust `_ensure_connection()` method. Evaluator needs the same pattern.

**Add after line 386:**
```python
async def _ensure_connection(self):
    """Create or verify RabbitMQ connection is alive.

    Mirrors Sampler._ensure_connection() for consistent reconnection behavior.
    """
    from disfun import process_utils

    needs_reconnect = (
        self.connection is None or
        self.connection.is_closed or
        self.channel is None or
        self.channel.is_closed
    )

    if needs_reconnect:
        logger.info(f"Evaluator {self.local_id}: Re-establishing RabbitMQ connection...")
        try:
            # Close stale connections
            await self._close_connection()

            # Create fresh connection
            self.connection = await process_utils.create_rabbitmq_connection(
                self._rabbitmq_config, timeout=60
            )
            self.channel = await self.connection.channel()
            self.evaluator_queue = await process_utils.declare_standard_queue(
                self.channel, "evaluator_queue"
            )
            self.database_queue = await process_utils.declare_standard_queue(
                self.channel, "database_queue"
            )
            logger.info(f"Evaluator {self.local_id}: Connection re-established")
            return True
        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Reconnection failed: {e}")
            return False
    return True

async def _close_connection(self):
    """Safely close existing connection."""
    try:
        if self.channel and not self.channel.is_closed:
            await self.channel.close()
    except Exception:
        pass
    try:
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
    except Exception:
        pass
    self.connection = None
    self.channel = None
    self.evaluator_queue = None
    self.database_queue = None
```

### 2.2 Update Evaluator `__init__` to Store Config

**File:** `src/disfun/evaluator.py:364`

**Add parameter and store it:**
```python
def __init__(self, connection, channel, evaluator_queue, database_queue, template,
             function_to_evolve, function_to_run, inputs, sandbox_base_path,
             timeout_seconds, local_id, target_signatures, max_workers=2,
             graph_dir=None, cache_graphs=False, cache_size_limit_gb=2.0,
             rabbitmq_config=None):  # ADD THIS
    # ... existing code ...
    self._rabbitmq_config = rabbitmq_config  # ADD THIS
```

### 2.3 Update Evaluator `consume_and_process()`

**File:** `src/disfun/evaluator.py:448-485`

**Replace with reconnection loop (like Sampler):**
```python
async def consume_and_process(self):
    """Main consume loop with automatic connection recovery."""
    reconnect_delay = 5.0
    max_reconnect_delay = 60.0

    while True:
        try:
            # Ensure connection is alive
            if self._rabbitmq_config is not None:
                connected = await self._ensure_connection()
                if not connected:
                    logger.error(f"Evaluator {self.local_id}: Connection failed, retrying in {reconnect_delay:.1f}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue

            # Reset delay on successful connection
            reconnect_delay = 5.0

            # Run consume loop
            await self._consume_loop()
            break

        except asyncio.CancelledError:
            logger.info(f"Evaluator {self.local_id}: Cancelled, exiting...")
            break

        except (aio_pika.exceptions.AMQPConnectionError,
                aio_pika.exceptions.ChannelClosed,
                ConnectionError, OSError) as e:
            logger.warning(f"Evaluator {self.local_id}: Connection error: {e}. Reconnecting...")
            await self._close_connection()
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Unexpected error: {e}", exc_info=True)
            await self._close_connection()
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

async def _consume_loop(self):
    """Inner consume loop - processes messages from queue."""
    async with self.channel:
        await self.channel.set_qos(prefetch_count=1)
        async with self.evaluator_queue.iterator() as stream:
            message_count = 0
            async for message in stream:
                async with message.process():
                    try:
                        await asyncio.wait_for(self.process_message(message), timeout=300)
                    except asyncio.TimeoutError:
                        logger.warning(f"Evaluator {self.local_id}: Message processing timed out")
                    except Exception as e:
                        logger.error(f"Evaluator {self.local_id}: Error processing: {e}")

                message_count += 1
                if message_count % 10 == 0:
                    killed = sandbox.cleanup_orphaned_sandbox_processes(logger)
                    if killed > 0:
                        logger.info(f"Cleaned up {killed} orphaned sandbox processes")
```

### 2.4 Update `process_entry.py` to Pass RabbitMQ Config to Evaluator

**File:** `src/disfun/process_entry.py`

Find where Evaluator is instantiated and add `rabbitmq_config=config.rabbitmq`.

---

## Phase 3: Fix Scaling Logic

### 3.1 Add Time-Based Disconnection Detection

**File:** `src/disfun/scaling_utils.py`

**Problem:** Current logic requires 2 consecutive checks (4+ minutes) to detect all samplers disconnected.

**Add to `__init__`:**
```python
def __init__(self, ...):
    # ... existing code ...
    self.sampler_zero_consumer_count = 0
    self.samplers_ever_connected = False
    self.last_sampler_activity_time = None  # NEW: Track when samplers were last active
    self.evaluator_zero_consumer_count = 0  # NEW: Also track evaluators
    self.last_evaluator_activity_time = None  # NEW
```

**Update `run_scaling_loop` (around line 283-310):**
```python
# Track activity time
current_time = time.time()

if sampler_consumer_count > 0:
    self.samplers_ever_connected = True
    self.sampler_zero_consumer_count = 0
    self.last_sampler_activity_time = current_time  # Update activity time
else:
    # No consumers - check if we should spawn replacement
    if self.samplers_ever_connected and sampler_message_count > 0:
        self.sampler_zero_consumer_count += 1

        # Calculate time since last activity
        time_since_activity = (current_time - self.last_sampler_activity_time
                               if self.last_sampler_activity_time else float('inf'))

        # Trigger replacement if:
        # 1. 2 consecutive checks with 0 consumers, OR
        # 2. Messages waiting for more than 2 minutes with 0 consumers
        should_spawn = (
            self.sampler_zero_consumer_count >= 2 or
            time_since_activity > 120  # 2 minutes timeout
        )

        if should_spawn:
            self.resource_logger.warning(
                f"ALERT: sampler_queue has {sampler_message_count} messages, "
                f"0 consumers for {self.sampler_zero_consumer_count} checks "
                f"({time_since_activity:.0f}s since last activity). Spawning replacement..."
            )
            # ... spawn logic ...
```

### 3.2 Fix Scale-Down to Check In-Flight Messages

**File:** `src/disfun/scaling_utils.py:269-272`

**Current (too aggressive):**
```python
elif evaluator_message_count == 0 and len(evaluator_processes) > min_evaluators:
    await self.terminate_process(evaluator_processes, "Evaluator")
```

**Change to:**
```python
# Only scale down if queue is empty AND has been empty for multiple checks
# This prevents killing evaluators while messages are in-flight
elif evaluator_message_count == 0:
    if not hasattr(self, '_evaluator_idle_count'):
        self._evaluator_idle_count = 0
    self._evaluator_idle_count += 1

    # Require 2 consecutive checks with empty queue before scaling down
    if self._evaluator_idle_count >= 2 and len(evaluator_processes) > min_evaluators:
        self.resource_logger.info("Queue empty for 2+ checks, scaling down evaluator")
        await self.terminate_process(evaluator_processes, "Evaluator")
        self._evaluator_idle_count = 0
else:
    self._evaluator_idle_count = 0  # Reset on non-empty queue
```

### 3.3 Add Bidirectional Sampler ID Sync

**File:** `src/disfun/scaling_utils.py:429-434`

**Current (one-way):**
```python
sampler_id = self.next_sampler_id
self.next_sampler_id += 1
if self.database is not None:
    self.database.next_sampler_id = self.next_sampler_id
```

**Add sync method:**
```python
def sync_from_database(self, database):
    """Sync counters from database (called after checkpoint load)."""
    if database is not None and hasattr(database, 'next_sampler_id'):
        if database.next_sampler_id > self.next_sampler_id:
            self.resource_logger.info(
                f"Syncing sampler ID from database: {self.next_sampler_id} -> {database.next_sampler_id}"
            )
            self.next_sampler_id = database.next_sampler_id
```

**Call this in `__main__.py` after loading checkpoint.**

### 3.4 Reduce Check Interval for Faster Response

**File:** `src/experiments/experiment1/config.py:257`

**Current:**
```python
check_interval: int = 60
```

**Recommendation:** Keep at 60s but the time-based detection (3.1) will catch issues faster.

---

## Phase 4: Fix Cleanup Logic

### 4.1 Fix Queue Deletion Race Condition

**File:** `src/disfun/__main__.py:1079-1105`

**Current (forces deletion):**
```python
await queue.delete(if_unused=False, if_empty=False)
```

**Change to:**
```python
# Wait for consumers to disconnect before deleting
for queue_name in ['evaluator_queue', 'sampler_queue', 'database_queue']:
    try:
        queue = await cleanup_channel.declare_queue(
            queue_name, durable=False, auto_delete=False, passive=True
        )

        # Check if queue has consumers
        queue_info = await cleanup_channel.declare_queue(
            queue_name, passive=True
        )

        if queue_info.consumer_count > 0:
            print(f"Queue {queue_name} has {queue_info.consumer_count} consumers, waiting...")
            await asyncio.sleep(2)  # Give consumers time to disconnect

        # Only delete if no consumers
        await queue.delete(if_unused=True, if_empty=False)
        print(f"Deleted queue: {queue_name}")
    except aio_pika.exceptions.ChannelNotFoundEntity:
        print(f"Queue {queue_name} does not exist, skipping")
    except Exception as e:
        print(f"Could not delete queue {queue_name}: {e}")
```

### 4.2 Add ResourceManager Cleanup

**File:** `src/disfun/scaling_utils.py`

**Add cleanup method:**
```python
def cleanup(self):
    """Clean up ResourceManager state (call during shutdown)."""
    self.resource_logger.info("ResourceManager: Cleaning up state...")

    # Clear tracking maps
    self.process_to_device_map.clear()
    self.process_start_times.clear()

    # Shutdown NVML if initialized
    if not self.cpu_only:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    self.resource_logger.info("ResourceManager: Cleanup complete")
```

**Call in `__main__.py._shutdown()` after cancelling scaling task:**
```python
# Cancel scaling task
if hasattr(main.task_manager, 'resource_manager'):
    main.task_manager.resource_manager.cleanup()
```

### 4.3 Unify Cleanup in process_utils.py

**File:** `src/disfun/process_utils.py`

The `graceful_shutdown` function should handle both Sampler and Evaluator consistently:

```python
async def graceful_shutdown(component_type: str, local_id: int, logger: logging.Logger,
                           loop, connection, channel, task, instance,
                           cleanup_done_flag: dict):
    """Shared graceful shutdown logic."""
    if cleanup_done_flag.get('done', False):
        return

    logger.info(f"{component_type} {local_id}: Initiating graceful shutdown...")

    # 1. Cancel consume task
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # 2. Cleanup instance (this handles reconnected connections)
    if instance:
        try:
            # Close current RabbitMQ connections (handles reconnected refs)
            if hasattr(instance, 'async_cleanup'):
                await instance.async_cleanup()
            elif hasattr(instance, '_close_connection'):
                await instance._close_connection()

            # Cleanup other resources (ProcessPoolExecutor, GPU, etc.)
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()
            if hasattr(instance, 'cleanup'):
                instance.cleanup()
        except Exception as e:
            logger.error(f"{component_type} {local_id}: Error during cleanup: {e}")

    # 3. Fallback: close original refs only if instance never handled them
    # (This is a safety net, instance cleanup should have handled it)

    cleanup_done_flag['done'] = True
    logger.info(f"{component_type} {local_id}: Graceful shutdown complete.")
    loop.stop()
```

---

## Phase 5: Integration Testing

### 5.1 Test Scenarios

Create test script at `src/disfun/tests/test_reconnection.py`:

```python
"""
Test scenarios for reconnection and scaling logic.

Run with: python -m pytest src/disfun/tests/test_reconnection.py -v
"""

async def test_sampler_reconnection():
    """Test that sampler reconnects after connection loss."""
    # 1. Start sampler
    # 2. Kill RabbitMQ connection (simulate network failure)
    # 3. Verify sampler reconnects within 60s
    # 4. Verify messages continue processing
    pass

async def test_evaluator_reconnection():
    """Test that evaluator reconnects after connection loss."""
    # 1. Start evaluator
    # 2. Kill RabbitMQ connection
    # 3. Verify evaluator reconnects
    pass

async def test_scaling_detects_dead_samplers():
    """Test that ResourceManager spawns replacement within 2 minutes."""
    # 1. Start sampler, verify connected
    # 2. Kill sampler process abruptly
    # 3. Add messages to sampler_queue
    # 4. Verify replacement spawned within 2 minutes
    pass

async def test_graceful_shutdown():
    """Test that shutdown properly closes all connections."""
    # 1. Start system with samplers and evaluators
    # 2. Send SIGTERM
    # 3. Verify all connections closed
    # 4. Verify queues are empty or deleted
    pass
```

### 5.2 Manual Testing Checklist

1. **Network Failure Test:**
   - Start experiment
   - Run: `sudo iptables -A OUTPUT -p tcp --dport 5672 -j DROP` (block RabbitMQ)
   - Wait 2 minutes
   - Verify samplers/evaluators log reconnection attempts
   - Run: `sudo iptables -D OUTPUT -p tcp --dport 5672 -j DROP` (restore)
   - Verify processes reconnect and resume

2. **RabbitMQ Restart Test:**
   - Start experiment
   - Run: `sudo systemctl restart rabbitmq-server`
   - Verify all processes reconnect within 2 minutes
   - Verify no duplicate samplers spawned

3. **Graceful Shutdown Test:**
   - Start experiment
   - Press Ctrl+C
   - Verify shutdown completes within 60 seconds
   - Verify no zombie processes: `ps aux | grep disfun`
   - Verify queues deleted: `rabbitmqctl list_queues`

---

## Implementation Order

| Priority | Task | Files | Estimated Effort |
|----------|------|-------|------------------|
| **0** | **Add `_shutdown_requested` flag to Sampler** | `sampler.py` | 10 min |
| **0** | **Add `request_shutdown()` method** | `sampler.py`, `evaluator.py` | 5 min |
| **0** | **Update `graceful_shutdown()` to set flag** | `process_utils.py` | 10 min |
| **0** | **Update Sampler consume loop to check flag** | `sampler.py` | 15 min |
| 1 | Reduce heartbeat to 60s | `process_utils.py` | 5 min |
| 2 | Log connection errors | `process_utils.py` | 5 min |
| 3 | Add `_ensure_connection()` to Evaluator | `evaluator.py` | 30 min |
| 4 | Update Evaluator consume loop (with flag checks) | `evaluator.py` | 30 min |
| 5 | Pass RabbitMQ config to Evaluator | `process_entry.py` | 10 min |
| 6 | Add time-based disconnection detection | `scaling_utils.py` | 20 min |
| 7 | Fix scale-down logic | `scaling_utils.py` | 15 min |
| 8 | Add ResourceManager.cleanup() | `scaling_utils.py` | 10 min |
| 9 | Fix queue deletion race | `__main__.py` | 15 min |
| 10 | Add bidirectional ID sync | `scaling_utils.py`, `__main__.py` | 15 min |
| 11 | Integration testing | Various | 1 hour |

**Total estimated time: ~4 hours**

### Critical Dependencies

```
Phase 0 (Shutdown Flag) ─────► Must be done FIRST
         │
         ▼
Phase 1 (Heartbeat) ──────────► Independent
         │
         ▼
Phase 2 (Evaluator Reconnection) ──► Depends on Phase 0 pattern
         │
         ▼
Phase 3 (Scaling) ────────────► Independent of above
         │
         ▼
Phase 4 (Cleanup) ────────────► Depends on Phase 0 for clean shutdown
```

---

## Config Changes Summary

Add to `config.py`:

```python
@dataclasses.dataclass(frozen=True)
class RabbitMQConfig:
    host: str = 'rabbitmq'
    port: int = 5672
    management_port: int = 15672
    username: str = 'guest'
    password: str = 'guest'
    vhost: str = ''
    heartbeat: int = 60  # NEW: Heartbeat interval (seconds)
    connection_timeout: int = 30  # NEW: Connection timeout (seconds)
```

---

## Rollback Plan

If issues occur after deployment:

1. **Revert heartbeat:** Set `heartbeat=172800` (temporary, not recommended long-term)
2. **Disable new reconnection:** Comment out `_ensure_connection()` calls
3. **Restore old scaling:** Revert `scaling_utils.py` changes

---

## Success Criteria

1. ✅ Network failures detected within 2 minutes (not 2 days)
2. ✅ Samplers and Evaluators reconnect automatically after network blip
3. ✅ ResourceManager spawns replacements within 2 minutes of failure
4. ✅ Graceful shutdown completes within 60 seconds
5. ✅ No duplicate samplers after reconnection
6. ✅ No zombie processes after shutdown
7. ✅ No queue deletion errors during shutdown
