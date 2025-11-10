#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=2
#SBATCH --mem=480GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=242
#SBATCH --gres=gpu:8
#SBATCH -o DeCoSearch/src/experiments/experiment1/logs/experiment.out
#SBATCH -e DeCoSearch/src/experiments/experiment1/logs/experiment.err
#SBATCH --time=48:00:00

# ===== Experiment config =====
EXPERIMENT_NAME="experiment1"
CONFIG_NAME="config.py"
RABBITMQ_CONF="rabbitmq.conf"
RABBITMQ_VHOST="${EXPERIMENT_NAME}"

PORT="15673"         # RabbitMQ mgmt HTTP
PORT2="5673"         # RabbitMQ AMQP
SSH_USER="ge74met"
SSH_HOST="login01.msv.ei.tum.de"
SSH_PORT="3022"

# ===== Node selection =====
NODE_SOURCE="${SLURM_JOB_NODELIST_HET_GROUP_0:-${SLURM_JOB_NODELIST}}"
mapfile -t NODE_LIST < <(scontrol show hostnames "$NODE_SOURCE")
NODE_1="${NODE_LIST[0]}"
REMAINING=("${NODE_LIST[@]:1}")

# ===== Resolve RabbitMQ host (FQDN of primary node) =====
RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist="$NODE_1" hostname -f)

# ===== Primary node: RabbitMQ & controller in container =====
srun -N1 -n1 --nodelist="$NODE_1" \
  --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
  --container-mounts="$PWD/DeCoSearch:/DeCoSearch,$PWD/.ssh:/DeCoSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_CONF="$RABBITMQ_CONF",RABBITMQ_VHOST="$RABBITMQ_VHOST",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",PORT="$PORT",PORT2="$PORT2",SSH_USER="$SSH_USER",SSH_HOST="$SSH_HOST",SSH_PORT="$SSH_PORT" \
  bash -s <<'REMOTE' &
set -euo pipefail

python3 /DeCoSearch/src/funsearchmq/update_config_file.py \
  "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}/${CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

export RABBITMQ_NODENAME="rabbit_${SLURM_JOB_ID}@localhost"
export RABBITMQ_USE_LONGNAME=true
export RABBITMQ_CONFIG_FILE="/DeCoSearch/src/experiments/${EXPERIMENT_NAME}/${RABBITMQ_CONF}"

rabbitmq-server &

sleep 30

curl -s -u guest:guest -X PUT "http://localhost:${PORT}/api/vhosts/${RABBITMQ_VHOST}"
curl -s -u guest:guest -X PUT \
  -H 'content-type: application/json' \
  -d '{"password":"mypassword","tags":"administrator"}' \
  "http://localhost:${PORT}/api/users/myuser"
curl -s -u guest:guest -X PUT \
  -H 'content-type: application/json' \
  -d '{"configure":".*","write":".*","read":".*"}' \
  "http://localhost:${PORT}/api/permissions/${RABBITMQ_VHOST}/myuser"

ssh -p "${SSH_PORT}" -N -f -R "${PORT}:localhost:${PORT}"  "${SSH_USER}@${SSH_HOST}"
ssh -p "${SSH_PORT}" -N -f -R "${PORT2}:localhost:${PORT2}" "${SSH_USER}@${SSH_HOST}"

cd /DeCoSearch
python3 -m pip install .

cd "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}"

python3 -m funsearchmq --sandbox_base_path "/mnt/sandboxstorage/${EXPERIMENT_NAME}" --checkpoint "/mnt/checkpoints/checkpoint_run_20251110_011412/checkpoint_2025-11-10_08-23-25.pkl"
REMOTE

# ===== Worker timing (tune as needed) =====
scaling_intervals_s=($(seq 180 200 360))   # sampler intervals
scaling_intervals_e=($(seq 100 30 300))   # evaluator intervals

# ===== Start evaluators & samplers only if there are extra nodes =====
if ((${#REMAINING[@]} > 0)); then
  sleep 300 # Wait for RabbitMQ to be fully up and for config update from primary node
  for i in "${!REMAINING[@]}"; do
    node="${REMAINING[$i]}"
    scaling_time_s=${scaling_intervals_s[$i]:-300}
    scaling_time_e=${scaling_intervals_e[$i]:-300}

    srun -N1 -n1 --nodelist="$node" \
      --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
      --container-mounts="$PWD/DeCoSearch:/DeCoSearch,$PWD/.ssh:/DeCoSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
      --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",scaling_time_s="$scaling_time_s",scaling_time_e="$scaling_time_e" \
      bash -s <<'REMOTE2' &
set -euo pipefail

python3 /DeCoSearch/src/funsearchmq/update_config_file.py \
  "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}/${CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

cd /DeCoSearch
python3 -m pip install .

cd "/DeCoSearch/src/experiments/${EXPERIMENT_NAME}"

python3 -u -m funsearchmq.attach_evaluators --check_interval="${scaling_time_e}" --sandbox_base_path "/mnt/sandboxstorage/${EXPERIMENT_NAME}" &
python3 -u -m funsearchmq.attach_samplers   --check_interval="${scaling_time_s}" &
wait
REMOTE2
  done
fi

wait  # Wait for backgrounded srun tasks