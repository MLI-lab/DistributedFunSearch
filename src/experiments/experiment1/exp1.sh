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
CONFIG_NAME="config.py"              # Config for main node (ProgramsDatabase)
WORKER_CONFIG_NAME="config.py"       # Config for worker nodes (attach_evaluators/attach_samplers)
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
  --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
  --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",RABBITMQ_CONF="$RABBITMQ_CONF",RABBITMQ_VHOST="$RABBITMQ_VHOST",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",PORT="$PORT",PORT2="$PORT2",SSH_USER="$SSH_USER",SSH_HOST="$SSH_HOST",SSH_PORT="$SSH_PORT" \
  bash -s <<'REMOTE' &

python3 /DistributedFunSearch/src/disfun/update_config_file.py \
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

# to connect ssh -p 3022 -L 15678:localhost:15673 ge74met@login01.msv.ei.tum.de -N -f
ssh -p "${SSH_PORT}" -N -f -R "${PORT}:localhost:${PORT}"  "${SSH_USER}@${SSH_HOST}"
ssh -p "${SSH_PORT}" -N -f -R "${PORT2}:localhost:${PORT2}" "${SSH_USER}@${SSH_HOST}"

# ===== RabbitMQ monitoring (background process) =====
RABBITMQ_LOG="/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/logs/rabbitmq_monitor.log"
(
  echo "$(date '+%Y-%m-%d %H:%M:%S'), RabbitMQ monitoring started" >> "$RABBITMQ_LOG"
  FAIL_COUNT=0
  MAX_FAILS=3
  while true; do
    # Restart RabbitMQ if unresponsive
    if ! curl -s -u guest:guest "http://localhost:${PORT}/api/overview" > /dev/null 2>&1; then
      ((FAIL_COUNT++))
      echo "$(date '+%Y-%m-%d %H:%M:%S'), RabbitMQ health check failed ($FAIL_COUNT/$MAX_FAILS)" >> "$RABBITMQ_LOG"
      if [ $FAIL_COUNT -ge $MAX_FAILS ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'), Restarting RabbitMQ..." >> "$RABBITMQ_LOG"
        pkill -f "rabbitmq-server" || true
        sleep 5
        rabbitmq-server &
        sleep 30
        # Re-create vhost and permissions after restart
        curl -s -u guest:guest -X PUT "http://localhost:${PORT}/api/vhosts/${RABBITMQ_VHOST}" || true
        curl -s -u guest:guest -X PUT -H 'content-type: application/json' \
          -d '{"configure":".*","write":".*","read":".*"}' \
          "http://localhost:${PORT}/api/permissions/${RABBITMQ_VHOST}/guest" || true
        FAIL_COUNT=0
        echo "$(date '+%Y-%m-%d %H:%M:%S'), RabbitMQ restarted" >> "$RABBITMQ_LOG"
      fi
    else
      FAIL_COUNT=0
    fi

    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$RABBITMQ_LOG"

    # Connection count and details
    echo "--- Connections ---" >> "$RABBITMQ_LOG"
    curl -s -u guest:guest "http://localhost:${PORT}/api/connections" 2>/dev/null | python3 -c "
import sys, json
try:
    conns = json.load(sys.stdin)
    print(f'Total connections: {len(conns)}')
    for c in conns[:10]:  # Show first 10
        print(f\"  {c.get('name', 'unknown')}: state={c.get('state')}, channels={c.get('channels')}, send_pend={c.get('send_pend', 0)}, recv_cnt={c.get('recv_cnt', 0)}\")
    if len(conns) > 10:
        print(f'  ... and {len(conns) - 10} more')
except: print('Failed to parse connections')
" >> "$RABBITMQ_LOG" 2>&1

    # Queue stats
    echo "--- Queues ---" >> "$RABBITMQ_LOG"
    curl -s -u guest:guest "http://localhost:${PORT}/api/queues" 2>/dev/null | python3 -c "
import sys, json
try:
    queues = json.load(sys.stdin)
    for q in queues:
        print(f\"  {q.get('name')}: messages={q.get('messages', 0)}, consumers={q.get('consumers', 0)}, memory={q.get('memory', 0)//1024}KB\")
except: print('Failed to parse queues')
" >> "$RABBITMQ_LOG" 2>&1

    # Node stats (memory, file descriptors, etc.)
    echo "--- Node Stats ---" >> "$RABBITMQ_LOG"
    curl -s -u guest:guest "http://localhost:${PORT}/api/nodes" 2>/dev/null | python3 -c "
import sys, json
try:
    nodes = json.load(sys.stdin)
    for n in nodes:
        mem_used = n.get('mem_used', 0) // (1024*1024)
        mem_limit = n.get('mem_limit', 0) // (1024*1024)
        fd_used = n.get('fd_used', 0)
        fd_total = n.get('fd_total', 0)
        sockets_used = n.get('sockets_used', 0)
        sockets_total = n.get('sockets_total', 0)
        proc_used = n.get('proc_used', 0)
        proc_total = n.get('proc_total', 0)
        print(f\"  Memory: {mem_used}MB / {mem_limit}MB ({100*mem_used//max(mem_limit,1)}%)\")
        print(f\"  File descriptors: {fd_used} / {fd_total} ({100*fd_used//max(fd_total,1)}%)\")
        print(f\"  Sockets: {sockets_used} / {sockets_total}\")
        print(f\"  Erlang processes: {proc_used} / {proc_total}\")
        if n.get('mem_alarm'): print('  WARNING: Memory alarm active!')
        if n.get('disk_free_alarm'): print('  WARNING: Disk free alarm active!')
except: print('Failed to parse node stats')
" >> "$RABBITMQ_LOG" 2>&1

    # Overview (message rates)
    echo "--- Overview ---" >> "$RABBITMQ_LOG"
    curl -s -u guest:guest "http://localhost:${PORT}/api/overview" 2>/dev/null | python3 -c "
import sys, json
try:
    o = json.load(sys.stdin)
    msg_stats = o.get('message_stats', {})
    publish_rate = msg_stats.get('publish_details', {}).get('rate', 0)
    deliver_rate = msg_stats.get('deliver_get_details', {}).get('rate', 0)
    ack_rate = msg_stats.get('ack_details', {}).get('rate', 0)
    print(f\"  Publish rate: {publish_rate:.1f}/s\")
    print(f\"  Deliver rate: {deliver_rate:.1f}/s\")
    print(f\"  Ack rate: {ack_rate:.1f}/s\")
    q_totals = o.get('queue_totals', {})
    print(f\"  Total messages: {q_totals.get('messages', 0)}\")
    print(f\"  Messages ready: {q_totals.get('messages_ready', 0)}\")
    print(f\"  Messages unacked: {q_totals.get('messages_unacknowledged', 0)}\")
except: print('Failed to parse overview')
" >> "$RABBITMQ_LOG" 2>&1

    echo "" >> "$RABBITMQ_LOG"
    sleep 30  # Log every 30 seconds
  done
) &
MONITOR_PID=$!
echo "RabbitMQ monitor started with PID $MONITOR_PID"

cd /DistributedFunSearch

echo "=== Installing disfun package ==="
python3 -m pip install . 2>&1 | tee "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/logs/pip_install.log"

echo "=== Checking vLLM installation ==="
python3 -c "import vllm; print('vLLM version:', vllm.__version__)" 2>&1 || {
  echo "vLLM not found, installing separately..."
  python3 -m pip install "vllm>=0.7.0,<0.8.0" 2>&1 | tee -a "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/logs/pip_install.log"
  python3 -c "import vllm; print('vLLM version:', vllm.__version__)" 2>&1 || echo "ERROR: vLLM installation failed"
}

cd "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}"

python3 -m disfun --sandbox_base_path "/tmp/sandbox_${EXPERIMENT_NAME}" --checkpoint "/mnt/checkpoints/checkpoint_run_20251126_014937_no_graph_input_seed1/checkpoint_2025-11-27_13-08-45.pkl"
REMOTE

# ===== Worker timing (tune as needed) =====
scaling_intervals_s=($(seq 180 200 360))   # sampler intervals
scaling_intervals_e=($(seq 100 30 300))   # evaluator intervals

# ===== Start evaluators & samplers only if there are extra nodes =====
if ((${#REMAINING[@]} > 0)); then
  for i in "${!REMAINING[@]}"; do
    node="${REMAINING[$i]}"
    scaling_time_s=${scaling_intervals_s[$i]:-300}
    scaling_time_e=${scaling_intervals_e[$i]:-300}

    srun -N1 -n1 --nodelist="$node" \
      --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
      --container-mounts="$PWD/DistributedFunSearch:/DistributedFunSearch,$PWD/.ssh:/DistributedFunSearch/.ssh,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/decosearch:/mnt" \
      --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",CONFIG_NAME="$CONFIG_NAME",WORKER_CONFIG_NAME="$WORKER_CONFIG_NAME",RABBITMQ_HOSTNAME="$RABBITMQ_HOSTNAME",scaling_time_s="$scaling_time_s",scaling_time_e="$scaling_time_e" \
      bash -s <<'REMOTE2' &

# Install disfun on worker node
cd /DistributedFunSearch
python3 -m pip install . --quiet 2>/dev/null || python3 -m pip install .
echo "Worker $(hostname): pip install complete"

# Wait for RabbitMQ to be ready on main node
sleep 120

python3 /DistributedFunSearch/src/disfun/update_config_file.py \
  "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/${WORKER_CONFIG_NAME}" "${RABBITMQ_HOSTNAME}"

cd "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}"

python3 -u -m disfun.attach_evaluators --config-path "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/${WORKER_CONFIG_NAME}" --check_interval="${scaling_time_e}" --sandbox_base_path "/tmp/sandbox_${EXPERIMENT_NAME}" &
python3 -u -m disfun.attach_samplers   --config-path "/DistributedFunSearch/src/experiments/${EXPERIMENT_NAME}/${WORKER_CONFIG_NAME}" --check_interval="${scaling_time_s}" &
wait
REMOTE2
  done
fi

wait  # Wait for backgrounded srun tasks