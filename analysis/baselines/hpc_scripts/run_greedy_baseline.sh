#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /dss/dsshome1/02/di38yur/DistributedFunSearch/analysis/baselines/logs/greedy_%j.out
#SBATCH -e /dss/dsshome1/02/di38yur/DistributedFunSearch/analysis/baselines/logs/greedy_%j.err
#SBATCH --time=24:00:00

# Configuration (override via environment variables)
GRAPH_TYPE=${GRAPH_TYPE:-deletion}
N_VALUES=${N_VALUES:-6,7,8,9,10}
S_VALUE=${S_VALUE:-1}
Q_VALUE=${Q_VALUE:-2}
TRIALS=${TRIALS:-100000}
SEED=${SEED:-42}
WORKERS=${WORKERS:-16}
GRAPH_DIR=${GRAPH_DIR:-/mnt/Graphs}
OUTPUT_DIR=${OUTPUT_DIR:-auto}
MEMORY_EFFICIENT=${MEMORY_EFFICIENT:-false}

# Create logs directory
mkdir -p logs

echo "============================================================"
echo "Random Greedy Baseline Job"
echo "============================================================"
echo "Type: $GRAPH_TYPE"
echo "n values: $N_VALUES"
echo "s: $S_VALUE"
echo "q: $Q_VALUE"
echo "Trials: $TRIALS"
echo "Seed: $SEED"
echo "Workers: $WORKERS"
echo "Graph dir: $GRAPH_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Memory efficient: $MEMORY_EFFICIENT"
echo "============================================================"

NODE_1=$(hostname -f)
echo "Running on: $NODE_1"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Build optional flags
EXTRA_FLAGS=""
if [ "$MEMORY_EFFICIENT" = "true" ]; then
    EXTRA_FLAGS="--memory-efficient"
fi

if [ "$OUTPUT_DIR" = "auto" ]; then
    OUTPUT_DIR="/DistributedFunSearch/analysis/baselines/greedy_results/${GRAPH_TYPE}_s${S_VALUE}_q${Q_VALUE}_$(date +%Y%m%d_%H%M%S)"
fi

srun -N1 -n1 \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/metis:/mnt" \
  bash -lc "
    set -euo pipefail
    echo 'Running on \$(hostname -f)'

    python3 -m venv /mnt/.venv 2>/dev/null || true
    source /mnt/.venv/bin/activate

    python3 -m pip install --upgrade pip -q
    python3 -m pip install python-Levenshtein tqdm lmdb ujson psutil -q

    cd /DistributedFunSearch/analysis/baselines

    echo 'Running random greedy baseline...'
    python3 random_greedy.py \\
        --n-values '$N_VALUES' \\
        --s $S_VALUE \\
        --q $Q_VALUE \\
        --graph-type '$GRAPH_TYPE' \\
        --graph-dir '$GRAPH_DIR' \\
        --trials $TRIALS \\
        --seed $SEED \\
        --workers $WORKERS \\
        --output '$OUTPUT_DIR' \\
        $EXTRA_FLAGS
  "

echo "============================================================"
echo "Job completed!"
echo "End time: $(date)"
echo "============================================================"
