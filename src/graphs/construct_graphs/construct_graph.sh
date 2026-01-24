#!/bin/bash
#SBATCH --partition=lrz-cpu                                          # Partition (queue) name
#SBATCH --qos=cpu
#SBATCH --nodes=1                                                    # Number of nodes
#SBATCH --mem=400GB                                                  # Memory per node
#SBATCH --ntasks-per-node=1                                          # Number of tasks per node
#SBATCH --cpus-per-task=92                                           # CPU cores per node
#SBATCH -o /dss/dsshome1/02/di38yur/DistributedFunSearch/src/graphs/construct_graphs/logs/experiment.out
#SBATCH -e /dss/dsshome1/02/di38yur/DistributedFunSearch/src/graphs/construct_graphs/logs/experiment.err
#SBATCH --time=48:00:00                                              # Time limit

# ==============================================================================
# Graph Construction Script
# ==============================================================================
#
# Usage examples (set these variables before submitting):
#
#   # Build deletion graphs for n=17,18,19 with s=2
#   GRAPH_TYPE=deletion N_VALUES=17,18,19 S_VALUE=2 Q_VALUE=2 WORKERS=90 sbatch construct_graph.sh
#
#   # Build IDS graphs for n=8,9,10 with s=1, quaternary
#   GRAPH_TYPE=ids N_VALUES=8,9,10 S_VALUE=1 Q_VALUE=4 WORKERS=90 sbatch construct_graph.sh
#
#   # Force streaming mode for memory-constrained systems
#   GRAPH_TYPE=deletion N_VALUES=19,20 S_VALUE=2 STREAM_MODE=stream sbatch construct_graph.sh
#
# ==============================================================================

# Configuration (override via environment variables)
GRAPH_TYPE=${GRAPH_TYPE:-deletion}    # "deletion" or "ids"
N_VALUES=${N_VALUES:-10,11,12}        # Comma-separated n values
S_VALUE=${S_VALUE:-1}                 # Number of errors to correct
Q_VALUE=${Q_VALUE:-2}                 # Alphabet size (2=binary, 4=DNA)
WORKERS=${WORKERS:-90}                # Number of parallel workers
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/Graphs} # Base output directory
STREAM_MODE=${STREAM_MODE:-auto}      # "auto", "stream", or "no-stream"

# Construct stream flag
STREAM_FLAG=""
if [ "$STREAM_MODE" = "stream" ]; then
    STREAM_FLAG="--stream"
elif [ "$STREAM_MODE" = "no-stream" ]; then
    STREAM_FLAG="--no-stream"
fi

echo "============================================================"
echo "Graph Construction Job"
echo "============================================================"
echo "Type: $GRAPH_TYPE"
echo "n values: $N_VALUES"
echo "s: $S_VALUE"
echo "q: $Q_VALUE"
echo "workers: $WORKERS"
echo "output: $OUTPUT_DIR"
echo "stream mode: $STREAM_MODE"
echo "============================================================"

# Get primary node
NODE_1=$(hostname -f)
echo "Running on: $NODE_1"

# Run graph construction
srun -N1 -n1 \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  bash -lc "
    set -euo pipefail
    echo 'Running on \$(hostname -f)'

    # Create a venv on a writable mount
    python3 -m venv /mnt/.venv 2>/dev/null || true
    source /mnt/.venv/bin/activate

    # Install dependencies
    python3 -m pip install --upgrade pip -q
    python3 -m pip install python-Levenshtein tqdm lmdb psutil -q

    cd /DistributedFunSearch/src/graphs/construct_graphs

    if [ '$GRAPH_TYPE' = 'deletion' ]; then
        echo 'Building deletion-correcting code graphs...'
        python3 construct_deletions_graphs.py \\
            --n '$N_VALUES' \\
            --s $S_VALUE \\
            --q $Q_VALUE \\
            --workers $WORKERS \\
            --output '$OUTPUT_DIR' $STREAM_FLAG
    elif [ '$GRAPH_TYPE' = 'ids' ]; then
        echo 'Building IDS (insertion/deletion/substitution) code graphs...'
        python3 construct_ids_graphs.py \\
            --n '$N_VALUES' \\
            --s $S_VALUE \\
            --q $Q_VALUE \\
            --workers $WORKERS \\
            --output '$OUTPUT_DIR' $STREAM_FLAG
    else
        echo 'Error: GRAPH_TYPE must be \"deletion\" or \"ids\"'
        exit 1
    fi
  "

echo "============================================================"
echo "Job completed!"
echo "============================================================"
