# Dynamic scaling configuration

DistributedFunSearch includes a dynamic scaling that automatically spawns and terminates sampler and evaluator processes based on queue load and available system resources.

## Overview

The scaling system monitors RabbitMQ queue load every `check_interval` seconds and makes decisions to:
- **Scale up**: Start new processes when queues are backed up and resources are available
- **Scale down**: Terminate idle processes when queues are empty

## Configuration

All scaling parameters are configurable via `ScalingConfig` in `config.py` or CLI arguments:

```python
config = Config(
    scaling=ScalingConfig(
        enabled=True,                        # Disable with --no-dynamic-scaling CLI flag
        check_interval=60,                   # Seconds between scaling checks
        max_samplers=1000,                   # Maximum samplers to scale up to
        max_evaluators=1000,                 # Maximum evaluators to scale up to

        # Queue thresholds (number of messages to trigger scale-up)
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

The `min_gpu_memory_gib` parameter should be adjusted based on the LLM's memory requirements.

## Scaling behavior

### Samplers

**Scale up when:**
1. `sampler_queue` message count > `sampler_scale_up_threshold`
2. Current sampler count < `max_samplers`
3. A GPU is available with:
   - Free memory ≥ `min_gpu_memory_gib` GiB
   - Utilization < `max_gpu_utilization`%
4. System free RAM ≥ `min_system_memory_gib` GiB

**Scale down when:**
- `sampler_queue` is empty (0 messages)

### Evaluators

**Scale up when (all are true):**
- `evaluator_queue` message count > `evaluator_scale_up_threshold`
- Current evaluator count < `max_evaluators`
- Smoothed average CPU usage < `cpu_usage_threshold`%
- Normalized load (1-min load avg / num cores) < `normalized_load_threshold`

**Scale down when:**
- `evaluator_queue` is empty (0 messages)

## Monitoring

Check the resource logs for scaling decisions:
```bash
tail -f logs/resources_<hostname>_pid<pid>.log
```
