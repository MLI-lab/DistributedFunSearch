#!/bin/bash
#SBATCH --job-name=disfun
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=250G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_%j.out
#SBATCH --error=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_%j.err



# ===== Configuration =====
EXPERIMENT_NAME="experiment1"
WORKSPACE="/data/horse/ws/frwe188h-disfun"
PROJECT_DIR="/home/frwe188h/DistributedFunSearch"
EXPERIMENT_DIR="${PROJECT_DIR}/src/experiments/${EXPERIMENT_NAME}"
RABBITMQ_VHOST="${EXPERIMENT_NAME}"

# Store large caches on horse (not home) - model weights can be 10-100+ GB
export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

mkdir -p "${WORKSPACE}/logs" "${WORKSPACE}/cache"

# ===== Node selection =====
mapfile -t NODE_LIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_1="${NODE_LIST[0]}"
REMAINING=("${NODE_LIST[@]:1}")

RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist="$NODE_1" hostname -f)
echo "RabbitMQ host: ${RABBITMQ_HOSTNAME}"

# Capture login node hostname for reverse tunnel
LOGIN_NODE=$(hostname -f)
echo "Login node: ${LOGIN_NODE}"

# ===== Primary node: RabbitMQ + Main experiment =====
srun -N1 -n1 --nodelist="$NODE_1" --exclusive bash -s <<REMOTE_PRIMARY &
set -e

# Load modules
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Store large caches on horse (not home), model weights can be 10-100+ GB
export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

# Activate pre-created venv
source "${WORKSPACE}/venv/bin/activate"

# Add Erlang to PATH (for RabbitMQ)
export PATH="/data/horse/ws/frwe188h-disfun/erlang/bin:\$PATH"

# Update config with RabbitMQ hostname and vhost
python3 "${PROJECT_DIR}/src/disfun/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOSTNAME}" "${RABBITMQ_VHOST}"

# Configure RabbitMQ to allow guest from any host (before starting)
mkdir -p /data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/etc/rabbitmq
cat > /data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/etc/rabbitmq/rabbitmq.conf << 'RMQCONF'
# Disable guest user loopback restriction
loopback_users.guest = false

# CRITICAL: Disable heartbeats to prevent false disconnects under high CPU load
# RabbitMQ Erlang VM can get CPU-starved and miss heartbeat frames
heartbeat = 0

# Performance tuning
vm_memory_high_watermark.relative = 0.6
frame_max = 524288
channel_max = 4096
tcp_listen_options.backlog = 4096
tcp_listen_options.sndbuf = 196608
tcp_listen_options.recbuf = 196608
collect_statistics_interval = 60000
RMQCONF

# Start RabbitMQ
/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmq-server &
sleep 30

# Enable management & setup vhost
/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmq-plugins enable rabbitmq_management

# Setup vhost and permissions
/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmqctl add_vhost "${RABBITMQ_VHOST}" 2>/dev/null || true
/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmqctl set_permissions -p "${RABBITMQ_VHOST}" guest ".*" ".*" ".*"

echo "RabbitMQ ready"

# Create reverse tunnel to login node for management access
ssh -N -f -R 15672:localhost:15672 ${LOGIN_NODE} 2>/dev/null || true
echo "Reverse tunnel to ${LOGIN_NODE}:15672 created"

# Run main experiment
cd "${EXPERIMENT_DIR}"
python3 -m disfun --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}"
REMOTE_PRIMARY

PRIMARY_PID=$!

# ===== Worker nodes =====
if ((${#REMAINING[@]} > 0)); then
    sleep 120  # Wait for RabbitMQ

    for i in "${!REMAINING[@]}"; do
        node="${REMAINING[$i]}"

        srun -N1 -n1 --nodelist="$node" --exclusive bash -s <<REMOTE_WORKER &
set -e

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Store large caches on horse (not home) - model weights can be 10-100+ GB
export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

source "${WORKSPACE}/venv/bin/activate"

python3 "${PROJECT_DIR}/src/disfun/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOSTNAME}"

cd "${EXPERIMENT_DIR}"
python3 -u -m disfun.attach_evaluators --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}" &
python3 -u -m disfun.attach_samplers &
wait
REMOTE_WORKER
    done
fi

wait $PRIMARY_PID
wait
