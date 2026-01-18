#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=2
#SBATCH --mem=150GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:3
#SBATCH -o /dss/dsshome1/02/di38yur/DistributedFunSearch/src/experiments/experiment1/logs/experiment.out
#SBATCH -e /dss/dsshome1/02/di38yur/DistributedFunSearch/src/experiments/experiment1/logs/experiment.err
#SBATCH --time=48:00:00

# ===== Experiment config =====
EXPERIMENT_NAME="experiment1"
CONFIG_NAME="config.py"
RABBITMQ_CONF="rabbitmq.conf"
RABBITMQ_VHOST="${EXPERIMENT_NAME}"
CHECKPOINT=""  # Set to resume from checkpoint, e.g. "/mnt/checkpoints/checkpoint_2025-01-01.pkl"

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
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_CONF="$RABBITMQ_CONF",RABBITMQ_VHOST="$RABBITMQ_VHOST",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",PORT="$PORT",PORT2="$PORT2",SSH_USER="$SSH_USER",SSH_HOST="$SSH_HOST",SSH_PORT="$SSH_PORT",CHECKPOINT="$CHECKPOINT" \
  bash -s <<'REMOTE' &

python3 /DistributedFunSearch/src/disfun/utils/update_config_file.py \
  "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/${CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

export RABBITMQ_NODENAME="rabbit_${SLURM_JOB_ID}@localhost"
export RABBITMQ_USE_LONGNAME=true
export RABBITMQ_CONFIG_FILE="/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/${RABBITMQ_CONF}"

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

ssh -p "${SSH_PORT}" -N -f -R "${PORT}:localhost:${PORT}" "${SSH_USER}@${SSH_HOST}"
ssh -p "${SSH_PORT}" -N -f -R "${PORT2}:localhost:${PORT2}" "${SSH_USER}@${SSH_HOST}"

cd /DistributedFunSearch
python3 -m pip install . 2>&1 | tee "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/logs/pip_install.log"

cd "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}"

if [[ -n "$CHECKPOINT" ]]; then
  python3 -m disfun --sandbox_base_path "/tmp/sandbox_${EXPERIMENT_NAME}" --checkpoint "$CHECKPOINT"
else
  python3 -m disfun --sandbox_base_path "/tmp/sandbox_${EXPERIMENT_NAME}"
fi
REMOTE

# ===== Worker nodes =====
if ((${#REMAINING[@]} > 0)); then
  for node in "${REMAINING[@]}"; do
    srun -N1 -n1 --nodelist="$node" \
      --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
      --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
      --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME" \
      bash -s <<'REMOTE2' &

cd /DistributedFunSearch
python3 -m pip install . --quiet 2>/dev/null || python3 -m pip install .

sleep 120  # Wait for RabbitMQ

python3 /DistributedFunSearch/src/disfun/utils/update_config_file.py \
  "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/${CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

cd "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}"

python3 -u -m disfun --attach evaluators --sandbox_base_path "/tmp/sandbox_${EXPERIMENT_NAME}" &
python3 -u -m disfun --attach samplers &
wait
REMOTE2
  done
fi

wait