#!/bin/bash
#SBATCH --partition=lrz-cpu                                # Partition (queue) name
#SBATCH --qos=cpu
#SBATCH --nodes=1                                                    # Number of nodes
#SBATCH --mem=400GB                                                  # Memory per node
#SBATCH --ntasks-per-node=1                                          # Number of tasks per node
#SBATCH --cpus-per-task=92                                           # CPU cores per node
#SBATCH -o /dss/dsshome1/02/di38yur/DeCoSearch/src/experiments/construct_graphs/logs2/experiment.out # Standard output log
#SBATCH -e /dss/dsshome1/02/di38yur/DeCoSearch/src/experiments/construct_graphs/logs2/experiment.err # Standard error log
#SBATCH --time=48:00:00                                              # Time limit


# Extract node lists for node groups
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)) || { echo "Error fetching node list"; exit 1; }

# Assign NODE_1 and remaining NODES
NODE_1=${NODE_LIST[0]} || { echo "Error assigning NODE_1"; exit 1; }
REMAINING=("${NODE_LIST[@]:1}")

echo "Primary node: $NODE_1"
echo "Remaining nodes: ${REMAINING[@]}"


# Get RabbitMQ hostname
RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist=$NODE_1 hostname -f) || { echo "Error getting RabbitMQ hostname"; exit 1; }
echo "RabbitMQ server hostname: $RABBITMQ_HOSTNAME"

# Run the main setup process on Node 1
srun -N1 -n1 --nodelist="$NODE_1" \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DeCoSearch:/DeCoSearch,$PWD/.ssh:/DeCoSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  bash -lc '
    set -euo pipefail
    echo "Running on $(hostname -f)"

    # Create a venv on a writable mount
    python3 -m venv /external/.venv
    source /external/.venv/bin/activate

    # Make sure pip exists/works, then install deps
    python3 -m pip install --upgrade pip
    python3 -m pip install python-Levenshtein tqdm lmdb psutil

    cd /DeCoSearch/src/construct_graphs
    python3 /DeCoSearch/src/construct_graphs/construct_ids_graphs.py
  '