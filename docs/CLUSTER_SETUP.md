# Cluster Setup Guide

Distributes work across multiple nodes. Node 1 runs RabbitMQ + ProgramsDatabase + initial workers. Remaining nodes attach additional samplers/evaluators.

## Method 1: Enroot Containers

For clusters with enroot/pyxis.

**Build container image** (once, on a compute node):

```bash
salloc -p <partition-name> --qos=<qos-name> --mem=64G && srun --pty bash

# Download PyTorch image
enroot import -o /path/to/disfun.sqsh docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install RabbitMQ inside container
enroot create -n disfun /path/to/disfun.sqsh
enroot start --root --rw disfun
apt update && apt install -y rabbitmq-server curl openssh-client build-essential
rabbitmq-plugins enable rabbitmq_management
exit
enroot export -o /path/to/disfun_rabbitmq.sqsh disfun
```

**Configure and submit:**

Edit `src/experiments/experiment1/run_enroot.sh` with your SLURM settings and container path, then:

```bash
sbatch DistributedFunSearch/src/experiments/experiment1/run_enroot.sh
```

---

## Method 2: Environment Modules

For clusters without container support.

**1. Create Python environment** (once):

```bash
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

python3 -m venv --system-site-packages /data/ws/your-workspace/venv
source /data/ws/your-workspace/venv/bin/activate
pip install --upgrade pip setuptools
pip install -e /path/to/DistributedFunSearch

# Build FastGraph C++ module (optional, for graph problems)
# This compiles for your active Python version (e.g., cpython-311 for Python 3.11).
# If nodes have different Python versions, build on each node separately.
# Requires: build-essential, liblmdb-dev (apt-get install -y build-essential liblmdb-dev)
cd /path/to/DistributedFunSearch
./src/disfun/utils/build_fast_graph.sh

wandb login
```

**2. Install RabbitMQ** (once):

```bash
# Erlang via conda
module load Anaconda3
conda create --prefix /data/ws/your-workspace/erlang erlang -c conda-forge -y

# RabbitMQ binary
wget https://github.com/rabbitmq/rabbitmq-server/releases/download/v3.13.3/rabbitmq-server-generic-unix-3.13.3.tar.xz
tar -xf rabbitmq-server-generic-unix-3.13.3.tar.xz && rm rabbitmq-server-generic-unix-3.13.3.tar.xz

# Test
export PATH="/data/ws/your-workspace/erlang/bin:$PATH"
./rabbitmq_server-3.13.3/sbin/rabbitmq-server --help
```

**3. Submit:**

```bash
sbatch src/experiments/experiment1/run_modules.sh
```

---

## How Workers Connect

Within a SLURM job, nodes can communicate directly. The `update_config_file.py` script updates `config.py` on each node, changing `host='localhost'` to the primary node's hostname.

Worker nodes then run:
```bash
python -m disfun --attach evaluators
python -m disfun --attach samplers
```

---

## Multi-Partition Setup

To run samplers on GPU partition and evaluators on CPU partition (separate SLURM jobs).

**Test connectivity first:**

```bash
# Terminal 1 (GPU partition)
salloc -p gpu-partition -N 1 --gres=gpu:1 && srun --pty bash
hostname -I                    # Note this IP
python -m http.server 5672     # Temporary listener

# Terminal 2 (CPU partition)
salloc -p cpu-partition -N 1 && srun --pty bash
python -c "import socket; s=socket.socket(); s.connect(('<gpu-ip>', 5672)); print('OK'); s.close()"
```

**If OK:** Set `host` in config to GPU node's hostname, then attach workers:

```bash
# Attach evaluators on CPU partition
sbatch --export=RABBITMQ_HOST=<gpu-node-hostname> src/experiments/experiment1/attach_cpus.sh

# Attach samplers on GPU partition
sbatch --export=RABBITMQ_HOST=<gpu-node-hostname> src/experiments/experiment1/attach_llms.sh
```

See `attach_cpus.sh` and `attach_llms.sh` for the full scripts.

**If fails:** Need SSH port forwarding through login node:
```bash
ssh -L 5672:<gpu-node>:5672 <gpu-node> -N &  # On login node
# Set host='login-node-hostname' in config
```

---

## SSH Tunnels

**RabbitMQ management UI** (from local machine):
```bash
ssh -L 15672:localhost:15672 user@cluster-login
# Open http://localhost:15672
```

**AMQP for external nodes:**
```bash
# On compute node: forward to external server
ssh -R 5672:localhost:5672 user@external-server -N -f
# External nodes connect to external-server:5672
```
