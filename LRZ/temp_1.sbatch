#!/bin/bash
#SBATCH -p mcml-hgx-h100-92x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=50GB 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5 # note could be that this is changing how many tasks are assigned per node (default is 1)      
#SBATCH --gres=gpu:1
#SBATCH -o Starcoder_Temp_1/enroot_test.out  
#SBATCH -e Starcoder_Temp_1/enroot_test.err  
#SBATCH --time=08:00:00  


# Get the list of allocated nodes
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST)) || { echo "Error fetching node list"; exit 1; }
echo "Allocated nodes: ${NODE_LIST[@]}"

NODE_1=${NODE_LIST[0]} || { echo "Error assigning NODE_1"; exit 1; }
REMAINING_NODES=(${NODE_LIST[@]:1})

# Get the fully qualified domain name (FQDN) of Node 1
RABBITMQ_HOSTNAME=$(srun -N1 -n1 --nodelist=$NODE_1 hostname -f) || { echo "Error getting RabbitMQ hostname"; exit 1; }
echo "RabbitMQ server hostname: $RABBITMQ_HOSTNAME"

# Run a Python script on Node 1 and log output to node1.log
srun --exclusive -N1 -n1 --nodelist=$NODE_1 \
     --container-mounts=/dss/dsshome1/02/di38yur/Starcoder_Temp_1:/workspace/main/,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/models/:/workspace/models/,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/Temp1/:/workspace/sandboxstorage/,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/code_backups/:/workspace/code_backups/\
     --container-image=/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh \
     bash -c "
         echo 'Running on $(hostname -f)'
         cd /workspace/main || { echo 'Error changing to /workspace/main'; exit 1; }

         # Set RabbitMQ node name inside the container to avoid conflicts
         export RABBITMQ_NODENAME=rabbit_${SLURM_JOB_ID}@localhost
         export RABBITMQ_USE_LONGNAME=true

         echo 'Updating config.py with RabbitMQ hostname on $RABBITMQ_HOSTNAME' || { echo 'Error updating config.py'; exit 1; }
         python /workspace/main/update_config_file.py /workspace/main/config.py \"$RABBITMQ_HOSTNAME\" || { echo 'Error running update_config_file.py'; exit 1; }

         pip install nvidia-ml-py3 || { echo 'Error installing nvidia-ml-py3'; exit 1; }
         pip install dash plotly numpy scikit-learn zss || { echo 'Error installing visual packages'; exit 1; }

         export RABBITMQ_CONFIG_FILE=/workspace/main/rabbitmq.conf
             
         # Ensure config and directories have the right permissions
         chmod 777 /workspace/main/rabbitmq.conf || { echo 'Error setting permissions on rabbitmq.conf'; exit 1; }

         # Start RabbitMQ with the custom config file
         rabbitmq-server || { echo 'Error starting RabbitMQ server'; exit 1; } &
             
         # Wait for RabbitMQ to fully start
         sleep 30 || { echo 'Error during sleep waiting for RabbitMQ'; exit 1; }

         # Create the virtual host (temp_1)
         curl -s -u guest:guest -X PUT http://localhost:15673/api/vhosts/temp_1 || { echo 'Error creating virtual host'; exit 1; }

         # Create a new RabbitMQ user after the server has started
         curl -s -u guest:guest -X PUT -d '{\"password\":\"mypassword\",\"tags\":\"administrator\"}' \
             -H 'content-type:application/json' http://localhost:15673/api/users/myuser || { echo 'Error creating RabbitMQ user'; exit 1; }

         # Set permissions for the new user on the virtual host
         curl -s -u guest:guest -X PUT -d '{\"configure\":\".*\", \"write\":\".*\", \"read\":\".*\"}' \
             -H 'content-type:application/json' http://localhost:15673/api/permissions/temp_1/myuser || { echo 'Error setting permissions'; exit 1; }
             
         echo 'RabbitMQ setup complete.'

         # Try to set up the reverse SSH tunnel for the management interface
         ssh -R 15673:localhost:15673 ge74met@login01.msv.ei.tum.de -p 3022 -N -f
         if [ $? -ne 0 ]; then
             echo 'Error setting up SSH tunnel for management interface, skipping visual.py'
         else
             # Run visual.py only if the SSH tunnel is successfully set up
             ssh -R 8052:localhost:8052 ge74met@login01.msv.ei.tum.de -p 3022 -N -f
             python /workspace/main/visual.py || { echo 'Error running visual.py'; } &
         fi

         # Ensure the directory has the right permissions
         chmod -R 750 /workspace/models/ || { echo 'Error setting directory permissions'; exit 1; }

         # Always run funsearch.py, regardless of whether the tunnels were successful or not
         python /workspace/main/funsearch.py || { echo 'Error running funsearch.py'; exit 1; }
     " &


# Create a list of 10 times evenly spaced from 1800 to 3600 seconds
scaling_intervals_s=($(seq 1800 200 3600))

# Create a list of times from 200 to 300 with a step of 30 seconds
scaling_intervals_e=($(seq 200 30 300))

sleep 120

# Run a Python script on Node 2 and log output to node2.log
for i in "${!REMAINING_NODES[@]}"; do
    node="${REMAINING_NODES[$i]}"
    # Get the scaling interval for the current node, wrapping around if more nodes than intervals
    scaling_time_s="${scaling_intervals_s[$i % ${#scaling_intervals_s[@]}]}"
    scaling_time_e="${scaling_intervals_e[$i % ${#scaling_intervals_e[@]}]}"

    srun --exclusive -N1 -n1 --nodelist=$node \
        --container-mounts=/dss/dsshome1/02/di38yur/Starcoder_Temp_1:/workspace/main/,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/models/:/workspace/models/,/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/Temp1/:/workspace/sandboxstorage/\
        --container-image=/dss/dssmcmlfs01/pn57vo/pn57vo-dss-0000/franziska/enroot/fw.sqsh \
        bash -c "
            echo 'Running on $(hostname -f)'
            cd /workspace/main || { echo 'Error changing to /workspace/main'; exit 1; }

            # Log the user running the script
            whoami || { echo 'Error checking user'; exit 1; }

            # Log the permissions and ownership of the directory
            ls -ld /workspace/models/ || { echo 'Error checking directory permissions'; exit 1; }

            # Ensure the directory has the right permissions, recursively
            chmod -R 750 /workspace/models/ || { echo 'Error setting directory permissions'; exit 1; }

            # Log the mount status
            mount | grep /workspace || { echo 'Error checking mount status'; exit 1; }

            pip install nvidia-ml-py3 || { echo 'Error installing nvidia-ml-py3 on Node 2'; exit 1; }

            # Run funsearch_e and funsearch_s in the background
            python /workspace/main/funsearch_e.py --check_interval_eval $scaling_time_e || { echo 'Error running funsearch_e.py'; exit 1; } &
            python /workspace/main/funsearch_s.py --check_interval_sam $scaling_time_s || { echo 'Error running funsearch_s.py'; exit 1; } &
            
            wait
        " &
done

wait  # Wait for all background tasks to complete
