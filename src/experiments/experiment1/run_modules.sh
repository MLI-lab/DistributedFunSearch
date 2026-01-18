#!/bin/bash
#SBATCH --job-name=disfun
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_%j.out
#SBATCH --error=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_%j.err

# ===== Configuration =====
EXPERIMENT_NAME="experiment1"
WORKSPACE="/data/horse/ws/frwe188h-disfun"
PROJECT_DIR="/home/frwe188h/DistributedFunSearch"
EXPERIMENT_DIR="${PROJECT_DIR}/src/experiments/${EXPERIMENT_NAME}"
CHECKPOINT=""  # Set to resume, e.g. "/mnt/checkpoints/checkpoint_2025-01-01.pkl"

RABBITMQ_PORT=5673
RABBITMQ_MANAGEMENT_PORT=15673
RABBITMQ_VHOST="${EXPERIMENT_NAME}"
RABBITMQ_CONF="${EXPERIMENT_DIR}/rabbitmq.conf"

export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

mkdir -p "${WORKSPACE}/logs" "${WORKSPACE}/cache"

# ===== Node selection =====
mapfile -t NODE_LIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_1="${NODE_LIST[0]}"
REMAINING=("${NODE_LIST[@]:1}")

RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist="$NODE_1" hostname -f)
LOGIN_NODE=$(hostname -f)

# ===== Primary node =====
srun -N1 -n1 --nodelist="$NODE_1" --exclusive bash -s <<REMOTE_PRIMARY &
set -e

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

source "${WORKSPACE}/venv/bin/activate"
export PATH="/data/horse/ws/frwe188h-disfun/erlang/bin:\$PATH"

RABBITMQ_BASE_DIR="${WORKSPACE}/rabbitmq/${SLURM_JOB_ID}"
mkdir -p "\${RABBITMQ_BASE_DIR}/mnesia" "\${RABBITMQ_BASE_DIR}/log"

python3 "${PROJECT_DIR}/src/disfun/utils/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOSTNAME}" "${RABBITMQ_VHOST}" "${RABBITMQ_PORT}" "${RABBITMQ_MANAGEMENT_PORT}"

mkdir -p /data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/etc/rabbitmq
cp "${RABBITMQ_CONF}" "/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/etc/rabbitmq/rabbitmq.conf"

export RABBITMQ_MNESIA_DIR="\${RABBITMQ_BASE_DIR}/mnesia"
export RABBITMQ_LOG_DIR="\${RABBITMQ_BASE_DIR}/log"

/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmq-server &
sleep 30

/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmq-plugins enable rabbitmq_management
/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmqctl add_vhost "${RABBITMQ_VHOST}" 2>/dev/null || true
/data/horse/ws/frwe188h-disfun/rabbitmq_server-3.13.3/sbin/rabbitmqctl set_permissions -p "${RABBITMQ_VHOST}" guest ".*" ".*" ".*"

ssh -N -f -R ${RABBITMQ_MANAGEMENT_PORT}:localhost:${RABBITMQ_MANAGEMENT_PORT} ${LOGIN_NODE} 2>/dev/null || true

cd "${EXPERIMENT_DIR}"
if [[ -n "${CHECKPOINT}" ]]; then
    python3 -m disfun --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}" --checkpoint "${CHECKPOINT}"
else
    python3 -m disfun --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}"
fi
REMOTE_PRIMARY

PRIMARY_PID=$!

# ===== Worker nodes =====
if ((${#REMAINING[@]} > 0)); then
    sleep 120

    for node in "${REMAINING[@]}"; do
        srun -N1 -n1 --nodelist="$node" --exclusive bash -s <<REMOTE_WORKER &
set -e

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

source "${WORKSPACE}/venv/bin/activate"

python3 "${PROJECT_DIR}/src/disfun/utils/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOSTNAME}" "${RABBITMQ_VHOST}" "${RABBITMQ_PORT}" "${RABBITMQ_MANAGEMENT_PORT}"

cd "${EXPERIMENT_DIR}"
python3 -u -m disfun --attach evaluators --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}" &
python3 -u -m disfun --attach samplers &
wait
REMOTE_WORKER
    done
fi

wait $PRIMARY_PID
wait
