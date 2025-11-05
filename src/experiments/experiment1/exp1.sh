#!/bin/bash
#SBATCH --partition=${PARTITION:-default}                            # Partition (queue) name
#SBATCH --nodes=5                                                    # Number of nodes
#SBATCH --mem=300GB                                                  # Memory per node
#SBATCH --ntasks-per-node=1                                          # Number of tasks per node
#SBATCH --cpus-per-task=92                                           # CPU cores per node
#SBATCH --gres=gpu:4                                                 # GPUs per node
#SBATCH -o DeCoSearch/src/experiments/experiment1/logs/experiment.out # Standard output log
#SBATCH -e DeCoSearch/src/experiments/experiment1/logs/experiment.err # Standard error log
#SBATCH --time=48:00:00                                              # Time limit

# Extract node lists for node groups
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)) || { echo "Error fetching node list"; exit 1; }

# Assign NODE_1 and remaining NODES
NODE_1=${NODE_LIST[0]} || { echo "Error assigning NODE_1"; exit 1; }
REMAINING=("${NODE_LIST[@]:1}")

echo "Primary node: $NODE_1"
echo "Remaining nodes: ${REMAINING[@]}"

# Experiment-specific variables
EXPERIMENT_NAME="experiment1"
CONFIG_NAME="config.py"
RABBITMQ_CONF="rabbitmq.conf"
PORT=15673
PORT2=5673
SSH_USER=" " # set user name
SSH_HOST="login01.msv.ei.tum.de"   # set to your laptop’s public IP or server name from which you want to access interface (note needs to be on same network as cluster)
SSH_PORT= #set to port at which to connect 


# Get RabbitMQ hostname
RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist=$NODE_1 hostname -f) || { echo "Error getting RabbitMQ hostname"; exit 1; }
echo "RabbitMQ server hostname: $RABBITMQ_HOSTNAME"

# Run the main setup process on Node 1
srun -N1 -n1 --nodelist=$NODE_1 \
     --container-image="/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh" \
     --container-mounts="$PWD/DeCoSearch:/DeCoSearch,\
$PWD/.ssh:/DeCoSearch/.ssh" \
     bash -c "
         echo 'Running on $(hostname -f)'

         # Update the RabbitMQ configuration with the hostname of allocated node
         python3 /DeCoSearch/src/decos/update_config_file.py /DeCoSearch/src/experiments/$EXPERIMENT_NAME/$CONFIG_NAME \"$RABBITMQ_HOSTNAME\" || { echo 'Error running update_config_file.py'; exit 1; }

         # Configure RabbitMQ environment
         export RABBITMQ_NODENAME=rabbit_${SLURM_JOB_ID}@localhost
         export RABBITMQ_USE_LONGNAME=true
         export RABBITMQ_CONFIG_FILE=/DeCoSearch/src/experiments/$EXPERIMENT_NAME/rabbitmq.conf

         # Start RabbitMQ in the foreground
         echo 'Starting RabbitMQ server...'
         rabbitmq-server &

         # Wait for RabbitMQ to fully start
         sleep 30 || { echo 'Error during sleep waiting for RabbitMQ'; exit 1; }

         # Create the virtual host
         curl -s -u guest:guest -X PUT http://localhost:$PORT/api/vhosts/exp1 || { echo 'Error creating virtual host'; exit 1; }

         # Create a new RabbitMQ user
         curl -s -u guest:guest -X PUT -d '{\"password\":\"mypassword\",\"tags\":\"administrator\"}' \
             -H 'content-type:application/json' http://localhost:$PORT/api/users/myuser || { echo 'Error creating RabbitMQ user'; exit 1; }

         # Set permissions for the new user on the virtual host
         curl -s -u guest:guest -X PUT -d '{\"configure\":\".*\", \"write\":\".*\", \"read\":\".*\"}' \
             -H 'content-type:application/json' http://localhost:$PORT/api/permissions/exp1/myuser || { echo 'Error setting permissions'; exit 1; }

         echo 'RabbitMQ setup complete.'

         # Set up reverse SSH tunnel for RabbitMQ management interface
         # Make sure to replace SSH_USER, SSH_HOST, and SSH_PORT with your actual SSH credentials
         # And keys are in .ssh folder for non-interactive login
         ssh -R $PORT:localhost:$PORT $SSH_USER@$SSH_HOST -p $SSH_PORT -N -f || { echo 'Error setting up SSH tunnel'; exit 1; }

         # Set up a reverse SSH tunnel for message passing, allowing external nodes to communicate with the main task running inside the cluster.
         ssh -R $PORT2:localhost:$PORT2  $SSH_USER@$SSH_HOST -p $SSH_PORT -N -f || { echo 'Error setting up SSH tunnel for RabbitMQ AMQP'; exit 1; }

         # Export API credentials (implementation is for an Azure-based API)

         # Install DeCoSearch
         cd /DeCoSearch
         python3 -m pip install .
        
         # Run DeCoSearch
         cd /DeCoSearch/src/experiments/$EXPERIMENT_NAME
         # Add command line arguments as needed
         python3 -m decos 
     " &

# Create a list of 10 times evenly spaced from 1800 to 3600 seconds
scaling_intervals_s=($(seq 180000 200 360000))
# Create a list of times from 200 to 300 with a step of 30 seconds
scaling_intervals_e=($(seq 200000 30 3000000))
sleep 120


# Run tasks on remaining nodes (evaluator and sampler scripts)
for i in "${!REMAINING[@]}"; do
    node="${REMAINING[$i]}"
    scaling_time_s=${scaling_intervals_s[$i]}  # Get scaling interval for sampler
    scaling_time_e=${scaling_intervals_e[$i]}  # Get scaling interval for evaluator
    srun -N1 -n1 --nodelist=$node \
     --container-image=desired/path/custom_name.sqsh \
     --container-mounts="$PWD/DeCoSearch:/DeCoSearch,\
$PWD/.ssh:/DeCoSearch/.ssh" \
        bash -c "
            echo 'Running on $(hostname -f)'

            # Update the RabbitMQ configuration with the hostname
            python /DeCoSearch/src/decos/update_config_file.py /DeCoSearch/src/experiments/$EXPERIMENT_NAME/$CONFIG_NAME \"$RABBITMQ_HOSTNAME\" || { echo 'Error running update_config_file.py'; exit 1; }

            # Install DeCoSearch
            cd /DeCoSearch
            pip install .

            cd /DeCoSearch/src/experiments/$EXPERIMENT_NAME

            python -m decos.attach_evaluators --check_interval=$scaling_time_e --sandbox_base_path=/workspace/sandboxstorage/ || { echo 'Error running attach_evaluators'; exit 1; } &

            python -m decos.attach_samplers --check_interval=$scaling_time_s || { echo 'Error running attach_sampler'; exit 1; } &
            wait
        " &
done

wait  # Wait for all tasks to complete

