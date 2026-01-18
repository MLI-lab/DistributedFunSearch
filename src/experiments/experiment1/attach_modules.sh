#!/bin/bash
#SBATCH --job-name=disfun-eval
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/evaluators_%j.out
#SBATCH --error=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/evaluators_%j.err

# ===== Configuration =====
EXPERIMENT_NAME="experiment1"
WORKSPACE="/data/horse/ws/frwe188h-disfun"
PROJECT_DIR="/home/frwe188h/DistributedFunSearch"
EXPERIMENT_DIR="${PROJECT_DIR}/src/experiments/${EXPERIMENT_NAME}"

RABBITMQ_PORT=5673
RABBITMQ_MANAGEMENT_PORT=15673
RABBITMQ_VHOST="${EXPERIMENT_NAME}"
CONNECTION_INFO_FILE="${EXPERIMENT_DIR}/rabbitmq_connection.txt"

export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"

# ===== Get RabbitMQ host =====
if [[ -n "$RABBITMQ_HOST" ]]; then
    echo "Using RABBITMQ_HOST: ${RABBITMQ_HOST}"
elif [[ -f "$CONNECTION_INFO_FILE" ]]; then
    source "$CONNECTION_INFO_FILE"
    echo "Loaded RABBITMQ_HOST from file: ${RABBITMQ_HOST}"
else
    echo "ERROR: RABBITMQ_HOST not set and ${CONNECTION_INFO_FILE} not found"
    echo "Usage: sbatch --export=RABBITMQ_HOST=<hostname> attach_modules.sh"
    exit 1
fi

# ===== Setup environment =====
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
source "${WORKSPACE}/venv/bin/activate"

python3 "${PROJECT_DIR}/src/disfun/utils/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "${RABBITMQ_HOST}" "${RABBITMQ_VHOST}" "${RABBITMQ_PORT}" "${RABBITMQ_MANAGEMENT_PORT}"

# ===== Wait for RabbitMQ =====
for i in $(seq 1 30); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('${RABBITMQ_HOST}', ${RABBITMQ_PORT})); s.close()" 2>/dev/null; then
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Cannot reach RabbitMQ at ${RABBITMQ_HOST}:${RABBITMQ_PORT}"
        exit 1
    fi
    echo "Waiting for RabbitMQ ($i/30)..."
    sleep 10
done

cd "${EXPERIMENT_DIR}"
python3 -u -m disfun --attach evaluators --sandbox_base_path "${WORKSPACE}/sandbox/${EXPERIMENT_NAME}"
