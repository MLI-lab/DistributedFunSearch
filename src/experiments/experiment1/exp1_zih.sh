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

# RabbitMQ ports (must match rabbitmq.conf)
RABBITMQ_PORT=5673
RABBITMQ_MANAGEMENT_PORT=15673
RABBITMQ_CONF="${EXPERIMENT_DIR}/rabbitmq.conf"

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

# Update config with RabbitMQ hostname, vhost, and ports
python3 "${PROJECT_DIR}/src/disfun/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOSTNAME}" "${RABBITMQ_VHOST}" "${RABBITMQ_PORT}" "${RABBITMQ_MANAGEMENT_PORT}"

# Configure RabbitMQ using the config file from the experiment directory
mkdir -p /data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/etc/rabbitmq
cp "${RABBITMQ_CONF}" /data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/etc/rabbitmq/rabbitmq.conf

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
ssh -N -f -R ${RABBITMQ_MANAGEMENT_PORT}:localhost:${RABBITMQ_MANAGEMENT_PORT} ${LOGIN_NODE} 2>/dev/null || true
echo "Reverse tunnel to ${LOGIN_NODE}:${RABBITMQ_MANAGEMENT_PORT} created"

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
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOSTNAME}" "${RABBITMQ_VHOST}" "${RABBITMQ_PORT}" "${RABBITMQ_MANAGEMENT_PORT}"

cd "${EXPERIMENT_DIR}"
python3 -u -m disfun.attach_evaluators --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}" &
python3 -u -m disfun.attach_samplers &
wait
REMOTE_WORKER
    done
fi

wait $PRIMARY_PID
wait
