#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /dss/dsshome1/02/di38yur/DistributedFunSearch/analysis/baselines/logs/gurobi_%j.out
#SBATCH -e /dss/dsshome1/02/di38yur/DistributedFunSearch/analysis/baselines/logs/gurobi_%j.err
#SBATCH --time=48:00:00

# Configuration (override via environment variables)
GRAPH_TYPE=${GRAPH_TYPE:-deletions}
N_VALUES=${N_VALUES:-10,11,12}
S_VALUE=${S_VALUE:-3}
Q_VALUE=${Q_VALUE:-2}
TIMEOUT=${TIMEOUT:-86400}
MIP_GAP=${MIP_GAP:-0}
GRAPH_DIR=${GRAPH_DIR:-/mnt/Graphs}
OUTPUT_DIR=${OUTPUT_DIR:-auto}
LOG_DIR=${LOG_DIR:-auto}

# Use SLURM allocated CPUs for Gurobi threads (or override with THREADS env var)
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-16}}

# Create logs directory
mkdir -p logs

echo "============================================================"
echo "Gurobi MIS Baseline Job"
echo "============================================================"
echo "Type: $GRAPH_TYPE"
echo "n values: $N_VALUES"
echo "s: $S_VALUE"
echo "q: $Q_VALUE"
echo "Timeout per instance: ${TIMEOUT}s ($((TIMEOUT/3600))h)"
echo "MIP gap: $MIP_GAP"
echo "Threads: $THREADS"
echo "Graph dir: $GRAPH_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo "============================================================"

NODE_1=$(hostname -f)
echo "Running on: $NODE_1"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Start time: $(date)"

if [ "$OUTPUT_DIR" = "auto" ]; then
    OUTPUT_DIR="/DistributedFunSearch/analysis/baselines/gurobi_results/${GRAPH_TYPE}_s${S_VALUE}_q${Q_VALUE}_$(date +%Y%m%d_%H%M%S)"
fi

if [ "$LOG_DIR" = "auto" ]; then
    LOG_DIR="${OUTPUT_DIR}/gurobi_logs"
fi

srun -N1 -n1 \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/metis:/mnt" \
  bash -lc "
    set -euo pipefail
    echo 'Running on \$(hostname -f)'

    echo '============================================================'
    echo 'System info:'
    echo 'CPUs available: '\$(nproc)
    free -h
    echo '============================================================'

    python3 -m venv /mnt/.venv 2>/dev/null || true
    source /mnt/.venv/bin/activate

    python3 -m pip install --upgrade pip -q
    python3 -m pip install gurobipy tqdm lmdb psutil -q

    # Check Gurobi license
    echo 'Checking Gurobi license...'
    python3 -c 'import gurobipy as gp; gp.Model()' 2>&1 && echo 'Gurobi license OK' || echo 'Gurobi license check failed'

    cd /DistributedFunSearch/analysis/baselines

    echo 'Running Gurobi MIS baseline...'
    python3 gurobi_mis.py \\
        --n-values '$N_VALUES' \\
        --s $S_VALUE \\
        --q $Q_VALUE \\
        --graph-type '$GRAPH_TYPE' \\
        --graph-dir '$GRAPH_DIR' \\
        --timeout $TIMEOUT \\
        --threads $THREADS \\
        --mip-gap $MIP_GAP \\
        --log-dir '$LOG_DIR' \\
        --output '$OUTPUT_DIR'

    echo '============================================================'
    echo 'Results saved to: $OUTPUT_DIR'
    echo 'Gurobi logs saved to: $LOG_DIR'
    ls -la '$OUTPUT_DIR' 2>/dev/null || true
    echo '============================================================'
  "

echo "============================================================"
echo "Job completed!"
echo "End time: $(date)"
echo "============================================================"
