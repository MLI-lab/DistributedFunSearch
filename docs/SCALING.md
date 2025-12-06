# Dynamic Scaling and Process Management

How DistributedFunSearch handles scaling, process lifecycle, network failures, and checkpoints.

## Dynamic Scaling

The ResourceManager monitors RabbitMQ queue depths every `check_interval` seconds and automatically spawns or terminates processes based on load and available resources.

### Configuration

All scaling parameters are configurable via `ScalingConfig` in `config.py`:

```python
config = Config(
    scaling=ScalingConfig(
        enabled=True,                        # Disable with --no-dynamic-scaling CLI flag
        check_interval=60,                   # Seconds between scaling checks
        max_samplers=1000,                   # Maximum samplers to scale up to
        max_evaluators=1000,                 # Maximum evaluators to scale up to

        # Queue thresholds (message count to trigger scale-up)
        sampler_scale_up_threshold=50,       # Messages in sampler_queue
        evaluator_scale_up_threshold=10,     # Messages in evaluator_queue

        # GPU resource thresholds (for samplers)
        min_gpu_memory_gib=20,               # Minimum free GPU memory (GiB)
        max_gpu_utilization=50,              # Maximum GPU utilization (%)

        # System resource thresholds
        min_system_memory_gib=30,            # Minimum free RAM (GiB)
        cpu_usage_threshold=99,              # Maximum CPU usage (%) for evaluators
        normalized_load_threshold=0.99,      # Maximum load-per-core for evaluators
    )
)
```

Adjust `min_gpu_memory_gib` based on your LLM's memory requirements (e.g., 30 GiB for StarCoder2-15B).

### Sampler Scaling

**Scale up when:**
1. `sampler_queue` message count > `sampler_scale_up_threshold`
2. Current sampler count < `max_samplers`
3. A GPU is available with free memory >= `min_gpu_memory_gib` and utilization < `max_gpu_utilization`
4. System free RAM >= `min_system_memory_gib`

**Scale down when:**
- `sampler_queue` empty for 2 consecutive checks
- Consumer count > `min_samplers`
- Selected sampler's GPU utilization < 30%

### Evaluator Scaling

**Scale up when:**
- `evaluator_queue` message count > `evaluator_scale_up_threshold`
- Current evaluator count < `max_evaluators`
- Smoothed CPU usage < `cpu_usage_threshold`
- Normalized load (1-min avg / cores) < `normalized_load_threshold`

**Scale down when:**
- `evaluator_queue` empty for 2 consecutive checks

### Disconnected Sampler Detection

If samplers disconnect but messages are waiting, ResourceManager detects this and spawns replacements:
- Triggers after 2 consecutive checks with 0 consumers but messages in queue
- Or after 2 minutes of no sampler activity with pending messages

## Process Lifecycle

### Startup

1. Main process spawns initial samplers via `torch.multiprocessing` (spawn context)
2. Main process spawns initial evaluators via `torch.multiprocessing` (spawn context)
3. Each process establishes its own RabbitMQ connection
4. Processes register signal handlers for graceful shutdown
5. ResourceManager begins monitoring queues and resources

### Graceful Shutdown

When a process receives SIGTERM or SIGINT:

1. **Set shutdown flag**: Prevents reconnection attempts
2. **Cancel consume task**: Stop processing new messages
3. **Finish in-flight work**: Allow current batch to complete
4. **Cleanup resources**:
   - Samplers: Kill vLLM child processes, release GPU memory
   - Evaluators: Shutdown ProcessPoolExecutor
5. **Close connections**: Close RabbitMQ channel and connection
6. **Exit**: Stop event loop

### Process Termination by ResourceManager

When scaling down:

1. **Select process**: Lowest GPU utilization (samplers) or oldest (evaluators)
2. **Minimum lifetime**: Process must run >= 60 seconds before eligible
3. **Idle check**: Queue must be empty for 2 consecutive checks
4. **Send SIGTERM**: Allow graceful shutdown (45s timeout for samplers, 30s for evaluators)
5. **Force kill**: SIGKILL if process doesn't exit in time
6. **Child cleanup**: Kill orphaned child processes (vLLM subprocesses)

### Dead Process Detection

ResourceManager's scaling loop automatically:
- Detects crashed processes via `is_alive()` check
- Removes dead processes from tracking
- Frees GPU assignments from dead samplers
- Allows replacement processes to be spawned

### vLLM Health Monitoring

Samplers detect hung vLLM instances:
- **Inference timeout**: 300 seconds (configurable via `inference_timeout`)
- **Consecutive failures**: After 5 failures, sampler exits for clean restart
- **GPU cleanup**: Child processes killed before exit to free GPU memory

## Network Failure Handling

### RabbitMQ Connection Management

All components (Sampler, Evaluator, ProgramsDatabase) use `RabbitMQConnectionManager` for consistent connection handling:

- **Robust connections**: Uses `aio_pika.connect_robust()` with automatic reconnection
- **Heartbeat**: Disabled by default (prevents false disconnects on busy clusters)
- **Connection validation**: Checked before each consume loop iteration
- **Reconnect backoff**: 5s initial delay, exponential backoff up to 60s max

### Queue Configuration

- **Non-durable**: `durable=False` (messages don't survive RabbitMQ restart)
- **Auto-delete disabled**: Queues persist when consumers disconnect
- **Consumer timeout**: 360000 seconds (100 hours) for slow consumers

### Message Acknowledgment

- Messages acknowledged only after successful processing
- Failed processing leaves message in queue for retry
- Prevents message loss on process crash

## Checkpoints

### Automatic Checkpointing

ProgramsDatabase saves checkpoints hourly:
- **Location**: `{checkpoints_base_path}/checkpoint_{timestamp}.pkl`
- **Contents**: Islands, clusters, programs, scores, counters, W&B run ID
- **Latest symlink**: `checkpoint_latest.pkl` points to most recent

### Resuming from Checkpoint

```bash
python -m disfun --checkpoint ./Checkpoints/checkpoint_latest.pkl
```

Restores:
- All program islands and clusters
- Score history and statistics
- W&B run (resumes same run)
- Sampler ID counter (ensures unique seeds across restarts)

## Sandbox Isolation

Generated functions execute in isolated sandboxes:

- **Separate process**: Each evaluation runs in subprocess via ProcessPoolExecutor
- **Timeout**: Configurable (default 90s) kills hung evaluations
- **Cleanup**: Sandbox directories cleaned after evaluation

## Monitoring

Check resource logs for scaling decisions:
```bash
tail -f logs/resources_<hostname>_pid<pid>.log
```

Check sampler/evaluator logs:
```bash
tail -f logs/samplers.log
tail -f logs/evaluators.log
```
