# Docker Setup Guide

FunSearchMQ uses **Docker Compose (v3.8)** to run two containers:

- **decos-main** (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`) - Runs the evolutionary search with GPU support
- **rabbitmq** (`rabbitmq:3.13.4-management`) - Message broker for asynchronous inter-process communication

Both containers run inside a **Docker bridge network** (`app-network`) for internal communication.

## Installation Steps

### 1. Start Docker Containers

```bash
cd .devcontainer
docker compose up --build -d
```

This will:
- Pull required Docker images (PyTorch with CUDA 12.1, RabbitMQ with management plugin)
- Create and start both containers in detached mode
- Set up the bridge network for container communication

### 2. Enter the Main Container

```bash
docker exec -it decos-main bash
```

### 3. Create Conda Environment

Inside the container, initialize conda and create the environment:

```bash
# Initialize conda for your shell (required inside Docker)
conda init bash
source ~/.bashrc

# Create and activate the environment
conda create -n env python=3.11 pip numpy==1.26.4 -y
conda activate env
```

### 4. Install PyTorch (Skip if Using API-based LLM)

If using a local LLM (e.g., StarCoder2):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If using OpenAI/Azure API (`sampler.gpt=True` in config), you can skip this step.

### 5. Install FunSearchMQ

```bash
cd /workspace/FunSearchMQ

# Standard installation
pip install .

# OR: Editable/development mode (changes to source code take effect immediately)
pip install -e .
```

### 6. (Optional) Pre-download LLM

To download StarCoder2-15B ahead of time:
```bash
cd src/experiments/experiment1
python load_llm.py  # Downloads to /workspace/models/
```

## Running an Experiment

```bash
cd src/experiments/experiment1
python -m funsearchmq
```

## Docker-Specific Configuration

### Network Architecture

- **Internal communication**: The main container connects to RabbitMQ via `rabbitmq:5672` (Docker service name)
- **External access**: RabbitMQ Management Interface is exposed on host port `15672`
- **Volume mounts**: Project directory is mounted at `/workspace/FunSearchMQ` (changes sync between host and container)

### RabbitMQ Connection

In `config.py`, use the Docker service name:
```python
rabbitmq=RabbitMQConfig(
    host='rabbitmq',  # Docker service name, not 'localhost'
    port=5672,
)
```

### Accessing RabbitMQ Management Interface

The RabbitMQ Management Interface is a web-based dashboard for monitoring message load, processing rates, and system status.

**Local access (same machine as Docker host):**
```
http://localhost:15672
```

**Remote access (Docker running on remote server):**

If running on a remote server, use SSH tunneling to access the interface on your local machine:

```bash
# Standard SSH tunnel
ssh -L 15672:localhost:15672 user@remote-server -N -f

# With jump server
ssh -J jump-user@jump-server -L 15672:localhost:15672 user@remote-server -N -f
```

Then access at `http://localhost:15672` on your local machine.

**Login credentials (default):**
- **Username**: guest
- **Password**: guest

### Running Multiple Experiments Simultaneously

To run multiple experiments in parallel without interference, use RabbitMQ virtual hosts (vhosts):

**1. Set different vhost for each experiment in `config.py`:**
```python
rabbitmq=RabbitMQConfig(
    host='rabbitmq',
    port=5672,
    vhost='exp1'  # Use 'exp2', 'exp3', etc. for other experiments
)
```

**2. Create the vhost in RabbitMQ:**
```bash
docker exec rabbitmq rabbitmqctl add_vhost exp1
docker exec rabbitmq rabbitmqctl set_permissions -p exp1 guest ".*" ".*" ".*"
```

**3. Repeat for each experiment:**
```bash
docker exec rabbitmq rabbitmqctl add_vhost exp2
docker exec rabbitmq rabbitmqctl set_permissions -p exp2 guest ".*" ".*" ".*"
```

Each experiment with a different `vhost` will be completely isolated with separate queues and messages.

### Modifying Ports or Hostnames

To customize ports or hostnames, edit `.devcontainer/docker-compose.yml`:

```yaml
services:
  rabbitmq:
    ports:
      - "15672:15672"  # Management interface
      - "5672:5672"    # AMQP message passing
```

## Troubleshooting

**Container can't connect to RabbitMQ:**
- Verify RabbitMQ is running: `docker ps`
- Check network connectivity: `docker exec decos-main ping rabbitmq`
- Ensure `host='rabbitmq'` (not `localhost`) in config

**GPU not available inside container:**
- Install NVIDIA Container Toolkit on host
- Verify GPU access: `docker exec decos-main nvidia-smi`

**Conda environment not persisting:**
- Add `conda activate env` to `~/.bashrc` inside container
- Or use `docker exec -it decos-main bash -c "conda activate env && bash"`
