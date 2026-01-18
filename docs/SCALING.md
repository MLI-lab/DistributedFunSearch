# Dynamic Scaling

ResourceManager monitors RabbitMQ queue depths and spawns/terminates processes based on load.

## Configuration

Configure via `ScalingConfig` in `config.py`:

```python
scaling=ScalingConfig(
    enabled=True,                     # Disable with --no-dynamic-scaling CLI flag
    check_interval=60,                # Seconds between scaling checks
    max_samplers=1000,
    max_evaluators=1000,
    sampler_scale_up_threshold=50,    # Queue depth to trigger scale-up
    evaluator_scale_up_threshold=10,
    min_gpu_memory_gib=20,            # For samplers (StarCoder2-15B needs ~30)
    max_gpu_utilization=50,
    cpu_usage_threshold=99,           # For evaluators
)
```

## Scaling Logic

**Samplers scale up** when queue > threshold, GPU available with sufficient memory, and system RAM available.

**Samplers scale down** when queue empty for 2 consecutive checks. Selects lowest GPU utilization.

**Evaluators scale up** when queue > threshold and CPU usage/load below thresholds.

**Evaluators scale down** when queue empty for 2 consecutive checks.

## Process Termination

1. Process must run >= 60s before eligible for termination
2. SIGTERM sent (45s timeout for samplers, 30s for evaluators)
3. SIGKILL if process doesn't exit
4. Orphaned vLLM child processes killed to free GPU memory

## RabbitMQ Connections

All components use `ConnectionManager` with `aio_pika.connect_robust()` for automatic reconnection. Exponential backoff from 5s to 60s max.

## Checkpoints

Saved hourly to `{checkpoints_base_path}/checkpoint_{timestamp}.pkl`.

Resume with:
```bash
python -m disfun --checkpoint ./Checkpoints/checkpoint_2024-01-15_12-30-00.pkl
```

## Monitoring

```bash
tail -f logs/resources_<hostname>_pid<pid>.log
```
