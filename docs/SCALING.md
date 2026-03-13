# Dynamic Scaling

ResourceManager monitors RabbitMQ queue depths and system resources to automatically spawn or terminate Samplers and Evaluators based on load.

## Configuration

```python
ScalingConfig(
    enabled=False,                        # Enable dynamic scaling (or use --dynamic-scaling CLI flag)
    resource_log_interval=60,             # Seconds between resource log entries

    # General
    check_interval=60,                    # Seconds between scaling checks
    max_samplers=1000,                    # Maximum sampler processes
    max_evaluators=200,                   # Maximum evaluator processes
    idle_checks_before_scale_down=2,      # Consecutive empty queue checks before scale-down
    min_process_lifetime=60,              # Seconds before process eligible for termination
    min_system_memory_gib=30,             # Min free RAM to spawn any process

    # Sampler scaling (GPU)
    sampler_scale_up_threshold=50,        # sampler_queue depth to trigger scale-up
    min_gpu_memory_gib=35,                # Min free GPU memory to start sampler
    max_gpu_utilization=50,               # Max GPU util % to allow scale-up
    min_gpu_util_for_scale_down=80,       # Skip scale-down if GPU util exceeds this
    sampler_termination_timeout=45,       # Seconds for graceful shutdown (vLLM cleanup)

    # Evaluator scaling (CPU)
    evaluator_scale_up_threshold=10,      # evaluator_queue depth to trigger scale-up
    cpu_usage_threshold=99,               # Max avg CPU % to allow scale-up
    normalized_load_threshold=0.99,       # Max load/cores to allow scale-up
    sample_count=10,                      # CPU samples to average
    sample_interval=1,                    # Seconds between CPU samples
    termination_timeout=30,               # Seconds for graceful shutdown
)
```

## Scaling Logic

### Samplers (GPU)

**Scale up** when:
- `sampler_queue` depth > `sampler_scale_up_threshold`
- GPU available with free memory > `min_gpu_memory_gib`
- GPU utilization < `max_gpu_utilization`
- System RAM > `min_system_memory_gib`

**Scale down** when:
- Queue empty for `idle_checks_before_scale_down` consecutive checks
- Process has run > `min_process_lifetime` seconds
- Selects sampler on GPU with lowest utilization (skips if > `min_gpu_util_for_scale_down`)

### Evaluators (CPU)

**Scale up** when:
- `evaluator_queue` depth > `evaluator_scale_up_threshold`
- CPU usage < `cpu_usage_threshold`
- Load/cores < `normalized_load_threshold`
- System RAM > `min_system_memory_gib`

**Scale down** when:
- Queue empty for `idle_checks_before_scale_down` consecutive checks
- Process has run > `min_process_lifetime` seconds

## Process Termination

1. SIGTERM sent with timeout (`sampler_termination_timeout` or `termination_timeout`)
2. SIGKILL if process doesn't exit
3. Orphaned vLLM child processes killed to free GPU memory

## Monitoring

Resource metrics logged to `logs/resources_<hostname>_pid<pid>.log` and wandb:

**CPU and memory:**
- `cpu_percent`, `cpu_cores_active`
- `memory_percent`, `memory_used_gib`, `memory_available_gib`
- `load_1min`, `load_5min`, `load_15min`, `load_per_core`
- `io_wait_percent`, `swap_percent`

**Disk and network:**
- `disk_read_mb`, `disk_write_mb`
- `net_sent_mb`, `net_recv_mb`

**GPU (per device):**
- `gpu_{idx}_utilization`, `gpu_{idx}_memory_util`
- `gpu_{idx}_memory_used_gib`, `gpu_{idx}_memory_free_gib`, `gpu_{idx}_memory_total_gib`
- `gpu_{idx}_temperature_c`, `gpu_{idx}_power_w`

**Queue depths:**
- `sampler_queue_depth`, `evaluator_queue_depth`, `database_queue_depth`

```bash
tail -f logs/resources_<hostname>_pid<pid>.log
```
