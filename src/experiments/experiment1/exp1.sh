#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:2
# NOTE: SBATCH lines do NOT expand shell variables.
#SBATCH -o DeCoSearch/src/experiments/experiment1/logs/experiment.out
#SBATCH -e DeCoSearch/src/experiments/experiment1/logs/experiment.err
#SBATCH --time=00:20:00

set -euo pipefail

# ===== Experiment config (vars DO expand below) =====
EXPERIMENT_NAME="experiment1"
CONFIG_NAME="config.py"
RABBITMQ_CONF="rabbitmq.conf"
RABBITMQ_VHOST="${EXPERIMENT_NAME}"   # vhost = experiment name

PORT="15673"   # RabbitMQ mgmt HTTP
PORT2="5673"   # RabbitMQ AMQP
SSH_USER="ge74met"
SSH_HOST="login01.msv.ei.tum.de"
SSH_PORT="3022"

# ===== Node selection (robust to non-hetero jobs) =====
NODE_SOURCE="${SLURM_JOB_NODELIST_HET_GROUP_0:-${SLURM_JOB_NODELIST:-}}"
if [[ -z "${NODE_SOURCE}" ]]; then
  echo "Error: SLURM_JOB_NODELIST is empty. Are you running under Slurm?"
  exit 1
fi

if ! mapfile -t NODE_LIST < <(scontrol show hostnames "$NODE_SOURCE"); then
  echo "Error fetching node list"
  exit 1
fi
if ((${#NODE_LIST[@]} == 0)); then
  echo "Error: node list resolved to zero nodes."
  exit 1
fi

NODE_1="${NODE_LIST[0]}"
REMAINING=("${NODE_LIST[@]:1}")

echo "Primary node: $NODE_1"
echo "Remaining nodes: ${REMAINING[*]:-(none)}"

# ===== Discover RabbitMQ host (fqdn of primary node) =====
RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist="$NODE_1" hostname -f) || { echo "Error getting RabbitMQ hostname"; exit 1; }
export RABBITMQ_HOSTNAME
echo "RabbitMQ server hostname: $RABBITMQ_HOSTNAME"

# ===== Primary node: start RabbitMQ & controller inside container =====
srun -N1 -n1 --nodelist="$NODE_1" \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DeCoSearch:/DeCoSearch,$PWD/.ssh:/DeCoSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/external" \
  --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_CONF="$RABBITMQ_CONF",RABBITMQ_VHOST="$RABBITMQ_VHOST",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",PORT="$PORT",PORT2="$PORT2",SSH_USER="$SSH_USER",SSH_HOST="$SSH_HOST",SSH_PORT="$SSH_PORT" \
  bash -s <<'REMOTE' &
set -euo pipefail

echo "Running on $(hostname -f)"

# Update the RabbitMQ hostname in your experiment config
python3 /DeCoSearch/src/funsearchmq/update_config_file.py \
  "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}/${CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

# RabbitMQ env
export RABBITMQ_NODENAME="rabbit_${SLURM_JOB_ID}@localhost"
export RABBITMQ_USE_LONGNAME=true
export RABBITMQ_CONFIG_FILE="/DeCoSearch/src/experiments/${EXPERIMENT_NAME}/${RABBITMQ_CONF}"

echo 'Starting RabbitMQ server...'
rabbitmq-server &

# Wait for mgmt API to come up
sleep 30

# Create vhost, user, and permissions via mgmt API
curl -s -u guest:guest -X PUT "http://localhost:${PORT}/api/vhosts/${RABBITMQ_VHOST}"
curl -s -u guest:guest -X PUT \
  -H 'content-type: application/json' \
  -d '{"password":"mypassword","tags":"administrator"}' \
  "http://localhost:${PORT}/api/users/myuser"
curl -s -u guest:guest -X PUT \
  -H 'content-type: application/json' \
  -d '{"configure":".*","write":".*","read":".*"}' \
  "http://localhost:${PORT}/api/permissions/${RABBITMQ_VHOST}/myuser"

echo 'RabbitMQ setup complete.'

# Reverse SSH tunnels (correct option order)
ssh -p "${SSH_PORT}" -N -f -R "${PORT}:localhost:${PORT}"  "${SSH_USER}@${SSH_HOST}"
ssh -p "${SSH_PORT}" -N -f -R "${PORT2}:localhost:${PORT2}" "${SSH_USER}@${SSH_HOST}"

# Install DeCoSearch
cd /DeCoSearch
python3 -m pip install .
echo 'Installed successfully.'

# Launch controller
cd "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}"
echo "In experiment directory: ${PWD}"
python3 -m funsearchmq
REMOTE

# ===== Worker timing (your values; keep or tune) =====
scaling_intervals_s=($(seq 180000 200 360000))   # sampler intervals
scaling_intervals_e=($(seq 200000 30 3000000))   # evaluator intervals

# ===== Start evaluators & samplers ONLY if there are extra nodes =====
if ((${#REMAINING[@]} > 0)); then
  sleep 120  # allow primary node to come up

  for i in "${!REMAINING[@]}"; do
    node="${REMAINING[$i]}"
    scaling_time_s=${scaling_intervals_s[$i]:-300}
    scaling_time_e=${scaling_intervals_e[$i]:-300}

    srun -N1 -n1 --nodelist="$node" \
      --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
      --container-mounts="$PWD/DeCoSearch:/DeCoSearch,$PWD/.ssh:/DeCoSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/external" \
      --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",scaling_time_s="$scaling_time_s",scaling_time_e="$scaling_time_e" \
      bash -s <<'REMOTE2' &
set -euo pipefail

echo "Running on $(hostname -f)"

# Ensure the experiment config points to the RabbitMQ host
# (If your repo uses funsearchmq vs decos here, keep it consistent with your codebase)
python3 /DeCoSearch/src/decos/update_config_file.py \
  "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}/${CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

# Install DeCoSearch
cd /DeCoSearch
python3 -m pip install .

cd "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}"

# Use the exported scaling vars from the outer shell
python3 -m decos.attach_evaluators --check_interval="${scaling_time_e}" --sandbox_base_path="/workspace/sandboxstorage/" &
python3 -m decos.attach_samplers   --check_interval="${scaling_time_s}" &
wait
REMOTE2
  done
else
  echo "No extra nodes detected; skipping worker launches."
fi

wait  # Wait for all backgrounded srun tasks