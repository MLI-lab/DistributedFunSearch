# Process Management

How DistributedFunSearch handles network failures, process lifecycle, and checkpoints.

## Network Failure Handling

### RabbitMQ Connection Management

All components use `RabbitMQConnectionManager` for automatic reconnection:

- **Robust connections**: Uses `aio_pika.connect_robust()` which automatically reconnects on network failures
- **Heartbeat**: Configurable heartbeat interval (disabled by default to prevent false disconnects with temporary network failure on clusters)
- **Connection validation**: `ensure_connection()` checks if connection is alive 
- **Automatic reconnection**: If connection drops, reconnect

### Retry Logic

- **Samplers/Evaluators**: Wrap consume loops with `with_reconnection()` for automatic retry
- **Initial delay**: 5 seconds, doubles on each failure up to 60 seconds max
- **Queue operations**: Failed publishes are retried after reconnection

## Process Lifecycle

### Startup

1. Main process spawns initial samplers and evaluators via `torch.multiprocessing`
2. Each process establishes its own RabbitMQ connection
3. Processes register signal handlers for graceful shutdown
4. ResourceManager begins monitoring queues and resources

### Graceful Shutdown

When a process receives SIGTERM or SIGINT:

1. **Stop consuming**: Cancel the message consume task
2. **Finish in-flight work**: Allow current message processing to complete
3. **Cleanup resources**:
   - Samplers: Unload LLM from GPU memory
   - Evaluators: Shutdown ProcessPoolExecutor
4. **Close connections**: Close RabbitMQ channel and connection
5. **Stop event loop**: Exit cleanly

### Process Termination by ResourceManager

When scaling down, ResourceManager:

1. **Selects process**: Chooses process with lowest GPU utilization (samplers) or oldest process
2. **Minimum lifetime**: Processes must run at least 60 seconds before termination
3. **Wait before terminating**: Queue must be empty for 2 consecutive scaling checks before terminating. This prevents killing a process while it's processing a message (queue appears empty during processing)
4. **Sends SIGTERM**: Allows graceful shutdown
5. **Timeout**: If process doesn't exit within 30 seconds, sends SIGKILL
6. **Child cleanup**: Also terminates child processes (vLLM may spawn subprocesses)

### Dead Process Cleanup

ResourceManager periodically:
- Removes dead processes from tracking lists
- Logs when processes crash unexpectedly
- Allows scaling system to spawn replacements

## Checkpoints

### Automatic Checkpointing

ProgramsDatabase saves checkpoints every hour:
- **Location**: `{checkpoints_base_path}/checkpoint_{timestamp}.pkl`
- **Contents**: All islands, clusters, programs, scores, counters, W&B run ID
- **Latest symlink**: `checkpoint_latest.pkl` always points to most recent

### Resuming from Checkpoint

```bash
python -m disfun --checkpoint ./Checkpoints/checkpoint_latest.pkl
```

Restores:
- All program islands and clusters
- Score history and statistics
- W&B run (resumes same run, not new one)
- Sampler ID counter (ensures unique seeds)

## Message Durability

### Queue Configuration

- **Durable queues**: Messages survive RabbitMQ restart
- **Auto-delete disabled**: Queues persist when consumers disconnect
- **Consumer timeout**: 360000 seconds (100 hours) to handle slow consumers

### Message Acknowledgment

- Messages are acknowledged only after successful processing
- Failed processing leaves message in queue for retry
- Prevents message loss on process crash

## Sandbox Isolation

Generated functions are executed in isolated sandboxes:

- **Separate process**: Each evaluation runs in subprocess via ProcessPoolExecutor
- **Timeout**: Configurable timeout (default 30s) kills hung evaluations
- **Cleanup**: Sandbox directories are cleaned after evaluation
- **No network**: Sandboxed code cannot make network requests
