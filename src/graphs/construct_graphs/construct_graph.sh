#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=<your-qos>
#SBATCH --nodes=1
#SBATCH --mem=400GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=92
#SBATCH -o <your-log-dir>/experiment.out
#SBATCH -e <your-log-dir>/experiment.err
#SBATCH --time=48:00:00

# Configuration (override via environment variables)
GRAPH_TYPE=${GRAPH_TYPE:-deletion}
N_VALUES=${N_VALUES:-10,11,12}
S_VALUE=${S_VALUE:-1}
Q_VALUE=${Q_VALUE:-2}
WORKERS=${WORKERS:-90}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/Graphs}
STREAM_MODE=${STREAM_MODE:-auto}

STREAM_FLAG=""
if [ "$STREAM_MODE" = "stream" ]; then
    STREAM_FLAG="--stream"
elif [ "$STREAM_MODE" = "no-stream" ]; then
    STREAM_FLAG="--no-stream"
fi

echo "type=$GRAPH_TYPE, n=$N_VALUES, s=$S_VALUE, q=$Q_VALUE, workers=$WORKERS, output=$OUTPUT_DIR, stream=$STREAM_MODE"

srun -N1 -n1 \
  --container-image="<your-container-image>" \
  --container-mounts="<your-repo>:/DistributedFunSearch,<your-data-dir>:/mnt" \
  bash -lc "
    set -euo pipefail

    python3 -m venv /mnt/.venv 2>/dev/null || true
    source /mnt/.venv/bin/activate

    python3 -m pip install --upgrade pip -q
    python3 -m pip install python-Levenshtein tqdm lmdb psutil -q

    cd /DistributedFunSearch/src/graphs/construct_graphs

    python3 construct_graphs.py \\
        --type '$GRAPH_TYPE' \\
        --n '$N_VALUES' \\
        --s $S_VALUE \\
        --q $Q_VALUE \\
        --workers $WORKERS \\
        --output '$OUTPUT_DIR' $STREAM_FLAG
  "
