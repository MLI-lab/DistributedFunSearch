# FunSearchMQ

<div align="center">
  <img src="fig/overview.png" alt="FunSearchMQ Overview" width="600">
</div>

<p>&nbsp;</p>

**FunSearchMQ** is a **distributed implementation of FunSearch** (Romera et al., 2024) that uses LLM-guided evolutionary search to discover novel algorithms. It supports multi-node execution via RabbitMQ for asynchronous message passing and works with both API-based LLMs (e.g., GPT-4o via Azure OpenAI) and locally hosted models (defaulting to StarCoder2).

## Features

FunSearchMQ is designed for large-scale distributed execution:

- **Multi-node execution**: Distributes across multiple nodes and allows adding workers to a running experiment from different nodes (see [Cluster Setup](docs/CLUSTER_SETUP.md) for SLURM/Enroot example)
- **Asynchronous workers**: ProgramsDatabase, Samplers, and Evaluators work independently
- **Dynamic scaling**: Automatically adjusts workers based on message load for maximum throughput (see [Scaling Guide](docs/SCALING.md))

In each iteration:

- A few-shot prompt is constructed by sampling from the program database, which stores all previously generated functions and their metadata.
- The LLM generates a new function variant.
- The function is evaluated on user-defined test cases.
- If the function is executable and logically distinct from previously stored ones, it is added to the program database along with its evaluation results.

Our implementation includes an example application for discovering large deletion-correcting codes. For details on this specific use case, see [our paper](https://arxiv.org/abs/2504.00613).

## Quickstart

### Installation

```bash
# 1. Create and activate conda environment
conda create -n env python=3.11 pip numpy==1.26.4 -y
conda activate env

# 2. Install PyTorch (skip if using API-based LLM)
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Start RabbitMQ
sudo systemctl start rabbitmq-server

# 4. Install FunSearchMQ
pip install .
```

See [Docker Setup](docs/DOCKER_SETUP.md) for container-based installation or [Cluster Setup](docs/CLUSTER_SETUP.md) for cluster execution.

### Run Experiment

This runs the example specification for discovering deletion-correcting codes:

```bash
cd src/experiments/experiment1
python -m funsearchmq

# Resume from checkpoint (if needed)
python -m funsearchmq --checkpoint path/to/checkpoint.pkl
```

## Modifying for Other Applications

Our implementation can be adapted to different applications with minimal changes:

**Input format and evaluation logic:**
Modify the input format of the function to be evolved in `src/experiments/experiment1/config.py` (via the `EvaluatorConfig` class). Set termination conditions using `TerminationConfig`:

```python
termination=TerminationConfig(
    prompt_limit=1_000_000,
    target_solutions={(7, 2): 5}  # Stop when target scores reached, or {} to disable
)
```

To adapt how functions are evaluated for your specific application, modify the logic in the `src/funsearchmq/specifications/` folder.

**LLM:**
Modify the `checkpoint` parameter in the sampler script (`src/funsearchmq/sampler.py`) to use a different open-source LLM that can be loaded from Hugging Face via `transformers.AutoModelForCausalLM`. For OpenAI models, set `sampler.gpt=True` in `config.py` and export your Azure OpenAI credentials.

**Configuration:**
All experiment parameters are configured in `config.py`. Key config blocks include:
- `PathsConfig`: Log and sandbox directories
- `ScalingConfig`: Dynamic scaling behavior (enable/disable, thresholds, resource limits)
- `TerminationConfig`: When to stop the experiment
- `WandbConfig`: Weights & Biases logging

See [Configuration Guide](docs/CONFIGURATION.md) for all configuration options.

## Documentation

- [Configuration Guide](docs/CONFIGURATION.md): Detailed configuration options, CLI arguments, and config blocks
- [Scaling Guide](docs/SCALING.md): Explanation on how dynamic resource scaling is implemented
- [Docker Setup](docs/DOCKER_SETUP.md): Setup with Docker containers
- [Cluster Setup](docs/CLUSTER_SETUP.md): Setup on cluster with SLURM and enroot 
