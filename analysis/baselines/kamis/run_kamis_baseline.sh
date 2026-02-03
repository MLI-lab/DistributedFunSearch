#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --mem=500GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -o /dss/dsshome1/02/di38yur/DistributedFunSearch/analysis/baselines/kamis/logs_s3_n18_online_mis_seed0/kamis_%j.out
#SBATCH -e /dss/dsshome1/02/di38yur/DistributedFunSearch/analysis/baselines/kamis/logs_s3_n18_online_mis_seed0/kamis_%j.err
#SBATCH --time=48:00:00

# Configuration (override via environment variables)
GRAPH_TYPE=${GRAPH_TYPE:-deletions}
N_VALUES=${N_VALUES:-18}
S_VALUE=${S_VALUE:-3}
Q_VALUE=${Q_VALUE:-2}
ALGORITHM=${ALGORITHM:-online_mis}
TIMEOUT=${TIMEOUT:-86400}  # 24h = 86400s or 169200
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
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw_v3.sqsh" \
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  bash -lc "
    set -euo pipefail
    echo 'Running on \$(hostname -f)'

    # Enable core dumps for debugging crashes
    ulimit -c unlimited
    echo 'Core dump limit: '\$(ulimit -c)

    # Show system memory info
    echo '============================================================'
    echo 'System Memory Info:'
    free -h
    echo '============================================================'

    python3 -m venv /mnt/.venv 2>/dev/null || true
    source /mnt/.venv/bin/activate

    python3 -m pip install --upgrade pip -q
    python3 -m pip install python-Levenshtein tqdm lmdb psutil -q

    cd /DistributedFunSearch/analysis/baselines/kamis

    chmod +x KaMIS/deploy/*

    # Quick sanity test of the binary
    echo '============================================================'
    echo 'Testing KaMIS binary with tiny graph...'
    echo '5 4
2
1 3
2 4
3 5
4' > /tmp/test_graph.metis
    if ./KaMIS/deploy/online_mis /tmp/test_graph.metis --console_log --time_limit=10 2>&1; then
        echo '[OK] Binary works on small graph'
    else
        echo '[WARN] Binary test failed with exit code: '\$?
    fi
    rm -f /tmp/test_graph.metis
    echo '============================================================'

    # Convert LMDB to METIS if needed (skips existing .metis files)
    echo 'Checking for METIS graphs (converting from LMDB if needed)...'
    python3 convert_lmdb_to_metis.py \\
        --n-values '$N_VALUES' \\
        --s $S_VALUE \\
        --q $Q_VALUE \\
        --graph-type '$GRAPH_TYPE' \\
        --graph-dir '$GRAPH_DIR'

    # Start background memory monitor
    echo 'Starting memory monitor in background...'
    (
        while true; do
            echo \"\$(date '+%Y-%m-%d %H:%M:%S') - Memory: \$(free -h | grep Mem | awk '{print \"Used: \"\$3\"/\"\$2}')\"
            sleep 300  # Log every 5 minutes
        done
    ) > /DistributedFunSearch/analysis/baselines/kamis/memory_monitor_\$\$.log 2>&1 &
    MONITOR_PID=\$!
    echo \"Memory monitor PID: \$MONITOR_PID\"

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

    EXIT_CODE=\$?

    # Stop memory monitor
    kill \$MONITOR_PID 2>/dev/null || true
    echo 'Memory monitor log:'
    cat /DistributedFunSearch/analysis/baselines/kamis/memory_monitor_\$\$.log 2>/dev/null || true

    # Post-run diagnostics
    echo '============================================================'
    echo 'Post-run diagnostics:'
    echo 'Exit code: '\$EXIT_CODE
    echo 'Memory after run:'
    free -h
    echo ''
    echo 'Checking for kernel errors (dmesg):'
    dmesg 2>/dev/null | tail -30 || echo '(dmesg not available in container)'
    echo ''
    echo 'Checking for core dumps:'
    ls -la core* 2>/dev/null || echo '(no core dumps found)'
    echo '============================================================'
  "

echo "============================================================"
echo "Job completed!"
echo "End time: $(date)"
echo "============================================================"
