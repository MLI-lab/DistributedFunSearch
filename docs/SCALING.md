# Dynamic Scaling Configuration Guide

DistributedFunSearch includes a dynamic scaling system that automatically spawns and terminates sampler and evaluator processes based on queue load and available system resources.

## Overview

The scaling system monitors RabbitMQ queue load every `check_interval` seconds (default: 120s) and makes decisions to:
- **Scale up**: Start new processes when queues are backed up and resources are available
- **Scale down**: Terminate idle processes when queues are empty

## Configuration

All scaling parameters are configurable via the `ScalingConfig` class in your experiment's `config.py`.

### ScalingConfig Parameters

```python
from src.experiments.experiment1.config import Config, ScalingConfig

config = Config(
    scaling=ScalingConfig(
        # Queue thresholds (number of messages to trigger scale-up)
        sampler_scale_up_threshold=50,      # Messages in sampler_queue
        evaluator_scale_up_threshold=10,    # Messages in evaluator_queue

        # GPU resource thresholds (for samplers)
        min_gpu_memory_gib=20,              # Minimum free GPU memory (GiB)
        max_gpu_utilization=50,              # Maximum GPU utilization (%)

        # System resource thresholds
        min_system_memory_gib=30,           # Minimum free RAM (GiB)
        cpu_usage_threshold=99,              # Maximum CPU usage (%) for evaluators
        normalized_load_threshold=0.99,      # Maximum load-per-core for evaluators
    )
)
```

The `min_gpu_memory_gib` parameter should be adjusted based on the LLM's memory requirements.


## Scaling Behavior

### Samplers

**Scale up when:**
1. `sampler_queue` message count > `sampler_scale_up_threshold` (default: 50)
2. Current sampler count < `max_samplers` (CLI argument, default: 1000)
3. A GPU is available with:
   - Free memory ≥ `min_gpu_memory_gib` GiB
   - Utilization < `max_gpu_utilization`%
4. System free RAM ≥ `min_system_memory_gib` GiB

**Scale down when:**
- `sampler_queue` is empty (0 messages)
- Current sampler count > `min_samplers` (CLI argument, default: 1)

### Evaluators

**Scale up when (all are true):**
- `evaluator_queue` message count > `evaluator_scale_up_threshold` (default: 10)
- Current evaluator count < `max_evaluators` (CLI argument, default: 1000)
- Smoothed average CPU usage < `cpu_usage_threshold`% (default: 99%)
- Normalized load (1-min load avg / num cores) < `normalized_load_threshold` (default: 0.99)

**Scale down when (all are true):**
- `evaluator_queue` is empty (0 messages)
- Current evaluator count > `min_evaluators` (CLI argument, default: 1)

## CLI Arguments

Additional scaling control via command-line arguments:

```bash
python -m disfun \
  --check_interval 120 \           # Scaling check interval (seconds)
  --max_evaluators 1000 \          # Maximum evaluators
  --max_samplers 1000 \            # Maximum samplers
  --no-dynamic-scaling             # Disable auto-scaling entirely
```

## Monitoring Scaling Activity

You can check the resource logs for scaling decisions:
```bash
tail -f logs/resources_<hostname>_pid<pid>.log
```