# DistributedFunSearch

<div align="center">
  <img src="fig/overview.png" alt="DistributedFunSearch Overview" width="600">
</div>

<p>&nbsp;</p>

**DistributedFunSearch** (`disfun`) is a **multi node distributed implementation of [FunSearch](https://github.com/google-deepmind/funsearch)** ([Romera et al., 2024](https://www.nature.com/articles/s41586-023-06924-6)) that uses LLM guided evolutionary search to discover novel algorithms. It uses RabbitMQ for asynchronous message passing and supports local models (via vLLM) and API models (via LiteLLM) through a unified inference interface.

- **Independent workers**: ProgramsDatabase, Samplers, and Evaluators work independently and process tasks asynchronously to maximize throughput
- **Multi node execution**: Distributes across multiple nodes and allows adding Samplers or Evaluators from the same or different nodes to a running experiment (see [Docker Setup](docs/DOCKER_SETUP.md), [Cluster Setup](docs/CLUSTER_SETUP.md)) 
- **Dynamic scaling**: Automatically spawns/terminates Samplers and Evaluators based on workload and available resources (see [Scaling Guide](docs/SCALING.md))

In each iteration:

1. A few-shot prompt is constructed by sampling high-scoring functions from the [program database](docs/PROGRAMS_DATABASE.md)
2. The LLM generates a new candidate function (optionally with metadata, e.g., a natural language summary)
3. The function is [evaluated](docs/EVALUATOR.md) in a sandboxed environment on user-defined problem instances
4. If executable and producing unique outputs (not a duplicate), the new function is stored with its scores and metadata

Our implementation includes an example application for discovering large deletion correcting codes. For details on this specific use case, see [our paper](https://arxiv.org/abs/2504.00613).

## Quickstart

### Installation

```bash
# 1. Create and activate conda environment
conda create -n env python=3.11 pip numpy==1.26.4 -y
conda activate env

# 2. Install PyTorch (skip if using API based LLM)
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install C compiler (required for vLLM/Triton)
# For local models only, skip if using API based LLM
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y

# 4. Start RabbitMQ
sudo systemctl start rabbitmq-server

# 5. Install DistributedFunSearch
pip install . # or pip install -e . for development mode

# 6. Build FastGraph C++ module (optional, for graph problems)
# NOTE: This compiles for your active Python version (e.g., cpython-310 for Python 3.10).
# If nodes have different Python versions, build on each node separately.
./tools/build_fast_graph.sh
```

See [Docker Setup](docs/DOCKER_SETUP.md) for container based installation or [Cluster Setup](docs/CLUSTER_SETUP.md) for cluster execution.

### Run experiment

This runs the example specification for discovering deletion correcting codes using StarCoder2-15B (local model):

```bash
cd src/experiments/experiment1
python -m disfun

# Resume from checkpoint (if needed)
python -m disfun --checkpoint path/to/checkpoint.pkl
```

**Note:** On first run, model weights (~30GB for StarCoder2-15B) are downloaded to `~/.cache/huggingface/`. To change the model or cache location, edit `model` or `cache_dir` in `src/experiments/experiment1/config.py`.

You can monitor message load in real time at `http://localhost:15672` (login: guest/guest). Enable the management plugin first with `sudo rabbitmq-plugins enable rabbitmq_management`.

## Evolve your problem

Adapt DistributedFunSearch to your problem by creating a specification directory with evaluation scripts, prompt templates, and initial functions. See `src/disfun/specifications/Deletions/` and its [README](src/disfun/specifications/Deletions/README.md) for a complete working example and detailed documentation.
