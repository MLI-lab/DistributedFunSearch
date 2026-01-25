#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/kamis_%j.out
#SBATCH -e logs/kamis_%j.err
#SBATCH --time=24:00:00

# Configuration (override via environment variables)
GRAPH_TYPE=${GRAPH_TYPE:-ids}
N_VALUES=${N_VALUES:-11}
S_VALUE=${S_VALUE:-1}
Q_VALUE=${Q_VALUE:-4}
ALGORITHM=${ALGORITHM:-redumis}
TIMEOUT=${TIMEOUT:-86400}  # 24h = 86400s
SEEDS=${SEEDS:-0}          # Single seed by default for 24h jobs
REDUMIS_CONFIG=${REDUMIS_CONFIG:-standard}
GRAPH_DIR=${GRAPH_DIR:-/mnt/Graphs}
OUTPUT_DIR=${OUTPUT_DIR:-auto}

# Calculate number of runs from seeds
NUM_RUNS=$(echo "$SEEDS" | tr ',' '\n' | wc -l)

# Create logs directory
mkdir -p logs

echo "============================================================"
echo "KaMIS Baseline Job"
echo "============================================================"
echo "Type: $GRAPH_TYPE"
echo "n values: $N_VALUES"
echo "s: $S_VALUE"
echo "q: $Q_VALUE"
echo "Algorithm: $ALGORITHM"
echo "Timeout per run: ${TIMEOUT}s ($((TIMEOUT/3600))h)"
echo "Seeds: $SEEDS"
echo "Number of runs: $NUM_RUNS"
echo "Graph dir: $GRAPH_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "============================================================"

NODE_1=$(hostname -f)
echo "Running on: $NODE_1"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

srun -N1 -n1 \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  bash -lc "
    set -euo pipefail
    echo 'Running on \$(hostname -f)'

    python3 -m venv /mnt/.venv 2>/dev/null || true
    source /mnt/.venv/bin/activate

    python3 -m pip install --upgrade pip -q
    python3 -m pip install python-Levenshtein tqdm lmdb psutil -q

    cd /DistributedFunSearch/analysis/baselines/kamis

    echo 'Running KaMIS baseline...'
    python3 kamis_baseline.py \\
        --n-values '$N_VALUES' \\
        --s $S_VALUE \\
        --q $Q_VALUE \\
        --graph-type '$GRAPH_TYPE' \\
        --graph-dir '$GRAPH_DIR' \\
        --algorithm $ALGORITHM \\
        --timeout $TIMEOUT \\
        --runs $NUM_RUNS \\
        --seeds '$SEEDS' \\
        --redumis-config $REDUMIS_CONFIG \\
        --output '$OUTPUT_DIR'
  "

echo "============================================================"
echo "Job completed!"
echo "End time: $(date)"
echo "============================================================"
