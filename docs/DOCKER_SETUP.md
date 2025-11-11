# Docker Setup Guide

DistributedFunSearch uses Docker Compose to run two containers: **disfun-main** (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`) for the evolutionary search with GPU support, and **rabbitmq** (`rabbitmq:3.13.4-management`) for message passing. Both containers communicate via a Docker bridge network.

**CUDA Compatibility:** The devcontainer uses PyTorch 2.2.2 with CUDA 12.1. Check your server's CUDA version with `nvidia-smi` and look for the version in the top-right corner. If it differs from 12.1, update the base image in `.devcontainer/Dockerfile` to match (e.g., `cuda11.8` or `cuda12.4`). Find compatible PyTorch Docker images [here](https://pytorch.org/get-started/previous-versions/).

## Quick Start

Start the containers from the `.devcontainer` directory:

```bash
cd .devcontainer
docker compose up --build -d
docker exec -it disfun-main bash
```

Inside the container, initialize conda and create the environment:

```bash
conda init bash && source ~/.bashrc
conda create -n env python=3.11 pip numpy==1.26.4 -y
conda activate env
```

Install PyTorch matching your CUDA version. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions find the matching installation command [here](https://pytorch.org/get-started/previous-versions/). You can skip this step if using OpenAI/Azure API.

Install DistributedFunSearch:

```bash
cd /workspace/DistributedFunSearch
pip install .  # or pip install -e . for development mode
```

Optionally pre-download the LLM (downloads to `/workspace/models/`):

```bash
cd src/experiments/experiment1
python load_llm.py  # Change cache_dir in load_llm.py to modify download location
```

If you don't preload, the model downloads to `/mnt/models/` by default (change in `src/disfun/sampler.py:79`).

Run an experiment:

```bash
cd src/experiments/experiment1
python -m disfun
```

## Configuration

The main container connects to RabbitMQ via the Docker service name. In your `config.py`, set `host='rabbitmq'` (not `localhost`):

```python
rabbitmq=RabbitMQConfig(host='rabbitmq', port=5672)
```

## RabbitMQ Management Interface

The web-based monitoring dashboard is enabled by default and available at `http://localhost:15672` with login credentials **guest/guest**.

If running on a remote server, the interface is not directly accessible from your local machine. Forward port 15672 using an SSH tunnel from your local machine:

```bash
# Standard SSH tunnel
ssh -L 15672:localhost:15672 user@remote-server -N -f

# With jump server
ssh -J jump-user@jump-server -L 15672:localhost:15672 user@remote-server -N -f
```

Then access at `http://localhost:15672` on your local machine and login with guest/guest.

## Running Multiple Experiments

To run parallel experiments without interference, use RabbitMQ virtual hosts. Set a different vhost in each experiment's `config.py` (e.g., `vhost='exp1'`, `vhost='exp2'`), then create the vhost and set permissions:

```bash
docker exec rabbitmq rabbitmqctl add_vhost exp1
docker exec rabbitmq rabbitmqctl set_permissions -p exp1 guest ".*" ".*" ".*"
```

Repeat for each experiment with different vhost names. Each experiment will have completely isolated queues.

## Multi-Node Setup

To scale across multiple machines, run RabbitMQ and the ProgramsDatabase on a main node, then attach additional samplers and evaluators from worker nodes. The main node uses `.devcontainer/` (runs both RabbitMQ and disfun-main), while worker nodes use `.devcontainer/external/.devcontainer/` (runs only disfun-main).

**Main node setup:**

Start both containers and run the full experiment (includes ProgramsDatabase, samplers, and evaluators):

```bash
cd /workspace/DistributedFunSearch/.devcontainer
docker compose up --build -d
docker exec -it disfun-main bash
# Follow installation steps, then:
cd src/experiments/experiment1
python -m disfun
```

**Worker node setup:**

Start the external devcontainer which uses `network_mode: "host"` to share the host's network:

```bash
cd /workspace/DistributedFunSearch/.devcontainer/external/.devcontainer
docker compose up --build -d
docker exec -it disfun-main bash
```

Inside the worker container, follow the installation steps above (conda env, PyTorch, DistributedFunSearch). In `config.py`, set the RabbitMQ host to the main node's actual hostname or IP address:

```python
rabbitmq=RabbitMQConfig(
    host='main-node-hostname',  # e.g., 'node1.cluster.com' or '192.168.1.10'
    port=5672,
)
```

Then attach only samplers and evaluators (don't run the full experiment, which would create a duplicate ProgramsDatabase):

```bash
cd src/experiments/experiment1

# Attach evaluators only
python -m disfun.attach_evaluators

# Or attach samplers only
python -m disfun.attach_samplers

# Or run both in separate terminals
```
