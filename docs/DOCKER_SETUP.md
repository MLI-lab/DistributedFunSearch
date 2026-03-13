# Docker setup guide

DistributedFunSearch uses Docker Compose to run two containers: **disfun-main** (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`) for the evolutionary search with GPU support, and **rabbitmq** (`rabbitmq:3.13.4-management`) for message passing. Both containers communicate via a Docker bridge network.

**CUDA Compatibility:** The devcontainer uses PyTorch 2.2.2 with CUDA 12.1. Check your server's CUDA version with `nvidia-smi`, update the base image in `.devcontainer/Dockerfile` to match (e.g., `cuda11.8` or `cuda12.4`). Find compatible PyTorch Docker images [here](https://pytorch.org/get-started/previous-versions/).

## Quick start

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

For other CUDA versions find the matching installation command [here](https://pytorch.org/get-started/previous-versions/). You can skip this step if using API models.

Install DistributedFunSearch:

```bash
cd /workspace/DistributedFunSearch
pip install .  # or pip install -e . for development mode
```

**Build FastGraph C++ module (optional, for graph problems):**

This compiles for your active Python version (e.g., cpython-311 for Python 3.11). If nodes have different Python versions, build on each node separately.

First install build dependencies:

```bash
apt-get update && apt-get install -y build-essential liblmdb-dev
```

Then build:

```bash
./tools/build_fast_graph.sh
```


**Install C compiler (required for local models):**

If using local models with vLLM, install gcc/g++ (required for Triton to compile CUDA kernels):

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
```

Before running the experiment, update your config to use the Docker RabbitMQ service name:

```bash
# Edit src/experiments/experiment1/config.py
# Change: host='localhost'
# To:     host='rabbitmq'
```

Run an experiment:

```bash
cd src/experiments/experiment1
python -m disfun
```

## RabbitMQ management interface

The web-based monitoring dashboard is enabled by default and available at `http://localhost:15672` with login credentials **guest/guest**.

If running on a remote server, the interface is not directly accessible from your local machine. Forward port 15672 using an SSH tunnel from your local machine:

```bash
# Standard SSH tunnel
ssh -L 15672:localhost:15672 user@remote-server -N -f

# With jump server
ssh -J jump-user@jump-server -L 15672:localhost:15672 user@remote-server -N -f
```

Then access at `http://localhost:15672` on your local machine and login with guest/guest.

## Running multiple experiments

To run parallel experiments without interference, use RabbitMQ virtual hosts. Set a different vhost in each experiment's `config.py` (e.g., `vhost='exp1'`, `vhost='exp2'`), then create the vhost and set permissions:

```bash
docker exec rabbitmq rabbitmqctl add_vhost exp1
docker exec rabbitmq rabbitmqctl set_permissions -p exp1 guest ".*" ".*" ".*"
```

Repeat for each experiment with different vhost names. Each experiment will have completely isolated queues.

## Multi-node setup

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

**Requirement:** Worker nodes must be able to reach the main node on port 5672 (RabbitMQ).

Get the main node's IP (run on main node):
```bash
hostname -I | awk '{print $1}'
```

Test connectivity (run on worker node):
```bash
python -c "import socket; s=socket.socket(); s.connect(('<main-node-ip>', 5672)); print('OK'); s.close()"
```

Start the external devcontainer which uses `network_mode: "host"` to share the host's network:

```bash
cd /workspace/DistributedFunSearch/.devcontainer/external/.devcontainer
docker compose up --build -d
docker exec -it disfun-main bash
```

Inside the worker container, follow the installation steps above (conda env, PyTorch, DistributedFunSearch). Update the RabbitMQ host to point to the main node:

```bash
# Edit src/experiments/experiment1/config.py
# Change: host='localhost'
# To:     host='192.168.1.10'  # Main node's IP or hostname
```

Then attach only samplers and evaluators (don't run the full experiment, which would create a duplicate ProgramsDatabase):

```bash
cd src/experiments/experiment1

# Attach evaluators only
python -m disfun --attach evaluators

# Or attach samplers only
python -m disfun --attach samplers
```
