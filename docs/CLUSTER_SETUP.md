# Cluster setup guide (SLURM + enroot)

The cluster setup distributes work across multiple nodes:
- **Node 1 (Primary)**: Runs RabbitMQ, ProgramsDatabase, and initial samplers/evaluators
- **Remaining Nodes**: Attach additional samplers and evaluators to scale processing

## Setup steps

Before building the enroot container, request an interactive compute node:

```bash
# Request a specific node
salloc -p lrz-cpu --qos=cpu --mem=64G --nodelist=cpu-009

# Or request any available node
salloc -p lrz-cpu --qos=cpu --mem=64G
```

Once the node is allocated, start an interactive shell:

```bash
srun --pty bash
```

You can now run the enroot commands below on the compute node to build the enroot image.

Download and convert a PyTorch image with the required CUDA version. For example, to install PyTorch 2.2.2 with CUDA 12.1:

```bash
enroot import -o /desired/path/custom_name.sqsh docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

Start the image with root privileges to install RabbitMQ, curl, and OpenSSH client:

```bash
enroot create -n custom_name /desired/path/custom_name.sqsh
enroot start --root --rw custom_name
apt update && apt install -y rabbitmq-server curl openssh-client
rabbitmq-plugins enable rabbitmq_management
```

Once the setup is complete, exit the container and save the changes in a new image:

```bash
exit
enroot export -o /desired/path/custom_name_with_rabbitmq.sqsh custom_name
```

You can now delete the original `custom_name` image. Use `custom_name_with_rabbitmq.sqsh` as your container image in `exp1.sh`.

## Configure experiment

Edit `src/experiments/experiment1/exp1.sh`:

```bash
# SLURM resources
#SBATCH --nodes=5              # Total nodes
#SBATCH --mem=300GB            # Memory per node
#SBATCH --cpus-per-task=92     # CPU cores per node
#SBATCH --gres=gpu:4           # GPUs per node
#SBATCH --time=48:00:00        # Time limit

# Set container image path
--container-image="/path/to/your/image.sqsh"

# SSH tunnel configuration (optional)
# Tunnel 1 (PORT): For accessing RabbitMQ management interface from your local computer for monitoring
# Tunnel 2 (PORT2): For allowing external nodes outside the cluster to communicate with RabbitMQ for message passing
SSH_USER="your_username"
SSH_HOST="your.server.com"
SSH_PORT=22
PORT=15673    # RabbitMQ management interface
PORT2=5672    # RabbitMQ AMQP message passing
```

## Submit SLURM job

From the parent directory containing `DistributedFunSearch/`:

```bash
sbatch DistributedFunSearch/src/experiments/experiment1/exp1.sh
```

**Note**: The job must be submitted from the directory containing the `DistributedFunSearch/` folder, not from inside it. The script mounts `$PWD/DistributedFunSearch:/DistributedFunSearch` into the container.

## How it works

The script automatically assigns roles:
1. **NODE_1**: Primary node running RabbitMQ and main experiment
2. **REMAINING**: Worker nodes running `attach_evaluators` and `attach_samplers`

### RabbitMQ setup (node 1)

The primary node starts the RabbitMQ server, creates a virtual host using the experiment name (e.g., `experiment1`), and runs the main experiment with `python -m disfun`. The `update_config_file.py` script updates `config.py` on each node, changing `RabbitMQConfig.host` from `'localhost'` to the actual hostname (e.g., `'node01.cluster.domain'`). Optionally, SSH tunnels can be created to access the RabbitMQ management interface remotely or to allow nodes from outside the cluster to attach samplers/evaluators.

### Worker nodes

Each remaining node updates `config.py` with the RabbitMQ hostname (main node's hostname) and attaches evaluators and samplers using `python -m disfun.attach_evaluators` and `python -m disfun.attach_samplers`. Both commands support the same CLI arguments as the main script. Each node by default uses different `--check_interval` values for staggered scaling decisions.

## SSH tunnel setup

The `exp1.sh` script sets up two reverse SSH tunnels from the cluster to an external server:

### Tunnel 1: RabbitMQ management interface (monitoring)
```bash
ssh -R $PORT:localhost:$PORT $SSH_USER@$SSH_HOST -p $SSH_PORT -N -f
```
- **Purpose**: Access the RabbitMQ management interface from your local computer for monitoring queue load and message flow
- **Usage**: On your external server, run:
  ```bash
  ssh -L PORT:localhost:PORT SSH_USER@SSH_HOST -p SSH_PORT
  ```
  Then access `http://localhost:PORT` (e.g., `http://localhost:15673`) in your browser

### Tunnel 2: RabbitMQ AMQP (external node communication)
```bash
ssh -R $PORT2:localhost:$PORT2 $SSH_USER@$SSH_HOST -p $SSH_PORT -N -f
```
- **Purpose**: Allow external nodes outside the cluster to attach samplers/evaluators and communicate with RabbitMQ
- **Usage**: External nodes can connect to RabbitMQ at `SSH_HOST:PORT2` (e.g., `external-server.com:5672`) for message passing

### SSH key setup for non-interactive login

The script requires passwordless SSH access to establish reverse tunnels. Set up SSH keys in the parent directory (where you run `sbatch` from):

**1. Create `.ssh` directory and generate keys:**
```bash
# In the parent directory containing DistributedFunSearch/
mkdir -p .ssh
chmod 700 .ssh
cd .ssh

# Generate SSH key pair (press Enter for all prompts to use defaults)
ssh-keygen -t ed25519 -f cluster_key -N ""
```

**2. Copy public key to external server:**
```bash
# Copy the public key to the server you want to tunnel to
ssh-copy-id -i cluster_key.pub -p SSH_PORT SSH_USER@SSH_HOST
```

**3. Create SSH config (optional but recommended):**
```bash
# Create .ssh/config file
cat > config << 'EOF'
Host tunnel_host
    HostName SSH_HOST
    User SSH_USER
    Port SSH_PORT
    IdentityFile ~/.ssh/cluster_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF

chmod 600 config
```

**4. Test the connection:**
```bash
# Test passwordless login works
ssh -i cluster_key -p SSH_PORT SSH_USER@SSH_HOST "echo Connection successful"
```

The script mounts the `.ssh` directory:
```bash
--container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh"
```

**Note**: The container sees the keys at `/DistributedFunSearch/.ssh/`, so SSH commands inside the container use that path automatically.


