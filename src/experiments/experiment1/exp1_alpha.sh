#!/bin/bash
#SBATCH --job-name=disfun-main
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_%j.out
#SBATCH --error=/home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_%j.err

# =============================================================================
# Alpha Node Script: RabbitMQ + Samplers (GPU)
#
# This runs on Alpha (GPU partition). Evaluators run separately on Barnard (CPU).
# Barnard reads rabbitmq_connection.txt to connect to RabbitMQ on this node.
# =============================================================================

# ===== Paths =====
EXPERIMENT_NAME="experiment1"
WORKSPACE="/data/horse/ws/frwe188h-disfun"
PROJECT_DIR="/home/frwe188h/DistributedFunSearch"
EXPERIMENT_DIR="${PROJECT_DIR}/src/experiments/${EXPERIMENT_NAME}"

# ===== RabbitMQ =====
RABBITMQ_PORT=5673
RABBITMQ_MANAGEMENT_PORT=15673
RABBITMQ_VHOST="${EXPERIMENT_NAME}"
RABBITMQ_CONF="${EXPERIMENT_DIR}/rabbitmq.conf"
RABBITMQ_SERVER="${WORKSPACE}/rabbitmq_server-3.13.3"

# ===== Setup directories =====
mkdir -p "${EXPERIMENT_DIR}/logs"

# ===== Get compute node info =====
mapfile -t NODE_LIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
COMPUTE_NODE="${NODE_LIST[0]}"
COMPUTE_HOSTNAME=$(srun -N1 -n1 --nodelist="$COMPUTE_NODE" hostname -f)

echo "=============================================="
echo "Alpha: RabbitMQ + Samplers"
echo "=============================================="
echo "Node: ${COMPUTE_HOSTNAME}"
echo "Job:  ${SLURM_JOB_ID}"
echo ""

# ===== Write connection info for Barnard evaluators =====
CONNECTION_FILE="${EXPERIMENT_DIR}/rabbitmq_connection.txt"
cat > "${CONNECTION_FILE}" <<EOF
RABBITMQ_HOST=${COMPUTE_HOSTNAME}
RABBITMQ_PORT=${RABBITMQ_PORT}
RABBITMQ_VHOST=${RABBITMQ_VHOST}
SLURM_JOB_ID=${SLURM_JOB_ID}
EOF
echo "Connection info: ${CONNECTION_FILE}"
echo ""

# ===== Run on compute node =====
srun -N1 -n1 --nodelist="$COMPUTE_NODE" --exclusive bash -s <<REMOTE_SCRIPT &
set -e

# --- Load modules ---
module --force purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0-CUDA-12.4.0 LMDB/0.9.31

# --- Environment ---
export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"
export PATH="${WORKSPACE}/erlang/bin:\$PATH"

source "${WORKSPACE}/venv/bin/activate"

# --- Build FastGraph C++ extension (local to /tmp) ---
echo "Building FastGraph C++ extension..."
CPP_SRC="${PROJECT_DIR}/src/disfun/utils/_fast_graph_cpp_src"
LOCAL_BUILD="/tmp/fastgraph_build_\${SLURM_JOB_ID}"
LOCAL_LIB="/tmp/fastgraph_lib_\${SLURM_JOB_ID}"

# Copy source to local /tmp and build there (avoids shared filesystem issues)
mkdir -p "\${LOCAL_BUILD}" "\${LOCAL_LIB}"
cp "${CPP_SRC}"/*.cpp "${CPP_SRC}"/*.hpp "${CPP_SRC}"/setup.py "\${LOCAL_BUILD}/"
cd "\${LOCAL_BUILD}"
echo "DEBUG: Building in \${LOCAL_BUILD}"
python3 setup.py build_ext --inplace
echo "DEBUG: Build done, checking for .so file..."
ls -la
cp fast_graph_cpp*.so "\${LOCAL_LIB}/"
echo "DEBUG: Copied to \${LOCAL_LIB}"
ls -la "\${LOCAL_LIB}/"
export FASTGRAPH_CPP_PATH="\${LOCAL_LIB}"
# Test C++ module
echo "DEBUG: Testing C++ module..."
cd "\${LOCAL_LIB}"
python3 -c 'from fast_graph_cpp import FastGraphCpp; print("FastGraph C++ loaded OK")'
echo "DEBUG: FastGraph build complete"

# --- Update config for this node ---
cd "${PROJECT_DIR}"
python3 "${PROJECT_DIR}/src/disfun/utils/update_config_file.py" \
    "${EXPERIMENT_DIR}/config.py" "localhost" "${RABBITMQ_VHOST}" "${RABBITMQ_PORT}" "${RABBITMQ_MANAGEMENT_PORT}"

# --- Setup RabbitMQ ---
RABBITMQ_DATA="/tmp/rabbitmq_\${SLURM_JOB_ID}"
mkdir -p "\${RABBITMQ_DATA}/mnesia" "\${RABBITMQ_DATA}/log"
mkdir -p "${RABBITMQ_SERVER}/etc/rabbitmq"
cp "${EXPERIMENT_DIR}/rabbitmq.conf" "${RABBITMQ_SERVER}/etc/rabbitmq/"

export RABBITMQ_MNESIA_DIR="\${RABBITMQ_DATA}/mnesia"
export RABBITMQ_LOG_DIR="\${RABBITMQ_DATA}/log"

echo "RabbitMQ data: \${RABBITMQ_DATA}"

# --- Debug: Check Erlang ---
echo "DEBUG: Checking Erlang..."
which erl || echo "ERROR: erl not found in PATH"
erl -version 2>&1 || echo "ERROR: erl -version failed"
echo "DEBUG: PATH=\${PATH}"

# --- Start RabbitMQ ---
echo "Starting RabbitMQ..."
echo "DEBUG: RABBITMQ_MNESIA_DIR=\${RABBITMQ_MNESIA_DIR}"
echo "DEBUG: RABBITMQ_LOG_DIR=\${RABBITMQ_LOG_DIR}"

# Start RabbitMQ - let output go to job log so we can see errors
"${RABBITMQ_SERVER}/sbin/rabbitmq-server" &
RABBITMQ_PID=\$!
echo "RabbitMQ started with PID \${RABBITMQ_PID}"

# Wait for RabbitMQ with retry loop (can take 60-120 seconds on first start)
echo "Waiting for RabbitMQ to start (this can take 60-120 seconds)..."
MAX_RETRIES=30
RETRY_DELAY=10
for i in \$(seq 1 \$MAX_RETRIES); do
    # Check if process is still running
    if ! kill -0 \${RABBITMQ_PID} 2>/dev/null; then
        echo "ERROR: RabbitMQ process died!"
        echo "Checking for logs in \${RABBITMQ_LOG_DIR}:"
        ls -la "\${RABBITMQ_LOG_DIR}/" 2>/dev/null || echo "No log directory"
        cat "\${RABBITMQ_LOG_DIR}/"*.log 2>/dev/null || echo "No log files"
        echo "Checking default log location:"
        ls -la "${RABBITMQ_SERVER}/var/log/rabbitmq/" 2>/dev/null || echo "No default logs"
        cat "${RABBITMQ_SERVER}/var/log/rabbitmq/"*.log 2>/dev/null || echo "No default log files"
        exit 1
    fi

    # Try to check status
    if "${RABBITMQ_SERVER}/sbin/rabbitmqctl" status > /dev/null 2>&1; then
        echo "RabbitMQ is ready! (took ~\$((i * RETRY_DELAY)) seconds)"
        break
    fi

    if [ \$i -eq \$MAX_RETRIES ]; then
        echo "ERROR: RabbitMQ failed to start after \$((MAX_RETRIES * RETRY_DELAY)) seconds"
        echo "Checking for logs in \${RABBITMQ_LOG_DIR}:"
        ls -la "\${RABBITMQ_LOG_DIR}/" 2>/dev/null || echo "No log directory"
        cat "\${RABBITMQ_LOG_DIR}/"*.log 2>/dev/null || echo "No log files"
        echo "Checking default log location:"
        ls -la "${RABBITMQ_SERVER}/var/log/rabbitmq/" 2>/dev/null || echo "No default logs"
        exit 1
    fi

    echo "  Waiting... (\${i}/\${MAX_RETRIES})"
    sleep \$RETRY_DELAY
done
echo "RabbitMQ broker started"

# Enable management plugin
"${RABBITMQ_SERVER}/sbin/rabbitmq-plugins" enable rabbitmq_management

# Extra wait for admin commands to be ready
sleep 5
echo "Verifying RabbitMQ accepts commands..."
"${RABBITMQ_SERVER}/sbin/rabbitmqctl" list_vhosts

# Setup vhost and permissions
"${RABBITMQ_SERVER}/sbin/rabbitmqctl" add_vhost "${RABBITMQ_VHOST}" 2>/dev/null || true
"${RABBITMQ_SERVER}/sbin/rabbitmqctl" set_permissions -p "${RABBITMQ_VHOST}" guest ".*" ".*" ".*"

echo ""
echo "=============================================="
echo "RabbitMQ ready - start Barnard evaluators now"
echo "=============================================="
echo ""

# --- Run experiment ---
echo "Starting disfun experiment..."
echo "Working directory: ${EXPERIMENT_DIR}"
cd "${EXPERIMENT_DIR}"
echo "Running: python3 -m disfun"
python3 -m disfun
echo "Experiment finished with exit code: \$?"

REMOTE_SCRIPT

wait $!

# Cleanup
rm -f "${CONNECTION_FILE}"
