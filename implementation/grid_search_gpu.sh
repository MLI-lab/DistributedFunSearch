#!/bin/bash

# List of remote servers
#REMOTE_SERVERS=("sequoia.mli.ei.tum.de" "gpumlp2.msv.ei.tum.de" "bigsur.mli.ei.tum.de" "gpumlp.msv.ei.tum.de" "zion.msv.ei.tum.de")
#REMOTE_SERVERS=("zion.msv.ei.tum.de")
REMOTE_SERVERS=("sequoia.mli.ei.tum.de" "bigsur.mli.ei.tum.de" "gpumlp.msv.ei.tum.de" "zion.msv.ei.tum.de")
REMOTE_SERVERS_GPU=("sequoia.mli.ei.tum.de" "bigsur.mli.ei.tum.de" "gpumlp.msv.ei.tum.de" "zion.msv.ei.tum.de")



# Docker container name
CONTAINER_NAME="pytorchFW"

# Paths to the Python scripts
SCRIPT_PATH_REMOTE_E="/franziska/Funsearch/implementation/funsearch_e.py"
SCRIPT_PATH_REMOTE_S="/franziska/Funsearch/implementation/funsearch_s.py"
ROOT_DIRECTORY="/franziska/Funsearch/implementation/"
GRID_PARAMS_FILE="/franziska/implementation/GridSearch/last_config_file.json"


# Timeout duration in seconds
TIMEOUT_DURATION=3630   #3630  

# Log file for remote output
REMOTE_LOG="remote.log"

# GPU memory and utilization thresholds
GPU_MEMORY_THRESHOLD=10000  # 10 GB
GPU_UTILIZATION_THRESHOLD=50  # 50%
TOTAL_MEMORY_REQUIRED=30000  # 30 GB

# Function to kill all Python processes on the main server with retries
kill_python_processes() {
    echo "$(date) - Killing all Python processes on the main server..." >> "$REMOTE_LOG"
    MAX_RETRIES=5
    RETRY_COUNT=0
    SUCCESS=false

    while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
        if pkill -f python; then
            echo "$(date) - All Python processes on the main server killed successfully." >> "$REMOTE_LOG"
            SUCCESS=true
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "$(date) - Failed to kill Python processes on the main server. Attempt $RETRY_COUNT of $MAX_RETRIES." >> "$REMOTE_LOG"
            sleep 2  # Wait for 2 seconds before retrying
        fi
    done

    if [[ $SUCCESS == false ]]; then
        echo "$(date) - Failed to kill Python processes on the main server after $MAX_RETRIES attempts." >> "$REMOTE_LOG"
    fi
}

# Function to restart Docker containers on remote servers
restart_remote_containers() {
    echo "$(date) - Restarting Docker containers on remote servers..." >> "$REMOTE_LOG"
    for SERVER in "${REMOTE_SERVERS[@]}"; do
        echo "$(date) - Restarting Docker container on $SERVER..." >> "$REMOTE_LOG"        
        ssh -n -o ConnectTimeout=10 "franziska@$SERVER" "
            docker restart $CONTAINER_NAME
        " >> "$REMOTE_LOG" 2>&1
        ssh_exit_status=$?
        if [ $ssh_exit_status -eq 0 ]; then
            echo "$(date) - Docker container on $SERVER restarted successfully." >> "$REMOTE_LOG"
        else
            echo "$(date) - Failed to restart Docker container on $SERVER. SSH exit status: $ssh_exit_status" >> "$REMOTE_LOG"
            continue  # Skip to the next server
        fi
    done
    echo "$(date) - Remote Docker containers restarted." >> "$REMOTE_LOG"
}        


# Function to update a single parameter in the remote config file (on the remote server and container)
update_remote_config_param() {
    local server=$1
    local param_name=$2
    local param_value=$3

    # Use SSH to run the update_config.py script inside the Docker container on the remote server
    ssh "franziska@$server" "
        docker exec $CONTAINER_NAME /bin/bash -c '
            cd .. &&
            cd $ROOT_DIRECTORY || { echo \"Failed to change to directory $ROOT_DIRECTORY on $server\"; exit 1; }
            python update_config.py config.py \"{\\\"$param_name\\\": $param_value}\" || { echo \"Failed to update config.py on $server\"; exit 1; }
            echo \"Successfully updated $param_name=$param_value in config.py on $server\";
        '
    " >> "$REMOTE_LOG" 2>&1

    if [ $? -ne 0 ]; then
        echo "$(date) - Failed to update $param_name on $server. Check the logs for details." >> "$REMOTE_LOG"
    else
        echo "$(date) - Successfully updated $param_name=$param_value on $server" >> "$REMOTE_LOG"
    fi
}


# Function to update the configuration on all remote servers
update_remote_configs() {
    local config_file="$1"
    local SERVER=$2

    # Get the list of parameter names from the JSON file
    param_names=$(jq -r 'keys[]' "$config_file")

    # Loop through the parameter names and update each one on the remote servers
    for param_name in $param_names; do
        param_value=$(jq -r --arg key "$param_name" '.[$key]' "$config_file")

        update_remote_config_param "$SERVER" "$param_name" "$param_value"
    done
}


# Function to check and delete RabbitMQ resources, including consumers
check_and_delete_rabbitmq_resources() {
    echo "$(date) - Checking for leftover RabbitMQ resources..." >> "$REMOTE_LOG"
    
    # Fetch and delete all queues
    echo "Fetching queues..." >> "$REMOTE_LOG"
    queues_json=$(curl -s -u guest:guest http://rabbitmqFW:15672/api/queues)
    echo "Fetched queues:" >> "$REMOTE_LOG"
    echo $queues_json | jq -c '.[] | .name' | while read queue; do
        queue=$(echo $queue | sed 's/"//g')  # Remove quotes from queue name
        encoded_queue=$(printf '%s' "$queue" | jq -sRr @uri)  # Encode queue name for URL
        echo "Deleting queue: $queue" >> "$REMOTE_LOG"
        response=$(curl -s -X DELETE -u guest:guest "http://rabbitmqFW:15672/api/queues/%2F/$encoded_queue" 2>&1)
        if [[ $response == *"Object Not Found"* ]]; then
            echo "Queue $queue not found, skipping..." >> "$REMOTE_LOG"
        else
            echo "Deleted queue: $queue" >> "$REMOTE_LOG"
        fi
    done

    echo "Fetching connections..." >> "$REMOTE_LOG"
    connections_json=$(curl -s -u guest:guest http://rabbitmqFW:15672/api/connections)
    echo $connections_json | jq -c '.[] | .name' | while read connection; do
        connection=$(echo $connection | sed 's/"//g')  # Remove quotes
        encoded_connection=$(printf '%s' "$connection" | jq -sRr @uri)  # Encode connection for URL
        echo "Closing connection: $connection" >> "$REMOTE_LOG"
        response=$(curl -s -X DELETE -u guest:guest "http://rabbitmqFW:15672/api/connections/$encoded_connection" 2>&1)
        if [[ $response == *"Object Not Found"* ]]; then
            echo "Connection $connection not found, skipping..." >> "$REMOTE_LOG"
        else
            echo "Closed connection: $connection" >> "$REMOTE_LOG"
        fi
    done

    # Fetch and close all channels
    echo "Fetching channels..." >> "$REMOTE_LOG"
    channels_json=$(curl -s -u guest:guest http://rabbitmqFW:15672/api/channels)
    echo $channels_json | jq -c '.[] | .name' | while read channel; do
        channel=$(echo $channel | sed 's/"//g')  # Remove quotes
        encoded_channel=$(printf '%s' "$channel" | jq -sRr @uri)  # Encode channel for URL
        echo "Closing channel: $channel" >> "$REMOTE_LOG"
        response=$(curl -s -X DELETE -u guest:guest "http://rabbitmqFW:15672/api/channels/$encoded_channel" 2>&1)
        if [[ $response == *"Object Not Found"* ]]; then
            echo "Channel $channel not found, skipping..." >> "$REMOTE_LOG"
        else
            echo "Closed channel: $channel" >> "$REMOTE_LOG"
        fi
    done

    echo "$(date) - RabbitMQ resources cleaned up." >> "$REMOTE_LOG"
}

# Function to check sampler queue and execute script if necessary
check_sampler_queue_and_execute() {
    echo "$(date) - Checking sampler queue and GPU resources on remote servers..." >> "$REMOTE_LOG"

    # Fetch the sampler queue information
    sampler_queue_info=$(curl -s --connect-timeout 10 --max-time 30 -s -u guest:guest http://rabbitmqFW:15672/api/queues | jq '.[] | select(.name == "sampler_queue")')

    if [ $? -ne 0 ]; then
        echo "$(date) - Failed to query RabbitMQ for queue info." >> "$REMOTE_LOG"
        return  # Return early and let the main loop continue
    fi

    # If the queue does not exist, create it
    if [ -z "$sampler_queue_info" ]; then
        echo "$(date) - The sampler_queue does not exist. Creating the queue..." >> "$REMOTE_LOG"
        response=$(curl -s -o /dev/null -w "%{http_code}" -X PUT -u guest:guest \
            -H "content-type:application/json" \
            -d '{"durable":false,"auto_delete":false}' \
            http://rabbitmqFW:15672/api/queues/%2F/sampler_queue)

        if [ "$response" -eq 201 ] || [ "$response" -eq 204 ]; then
            echo "$(date) - Successfully created sampler_queue." >> "$REMOTE_LOG"
        else
            echo "$(date) - Failed to create sampler_queue. HTTP Status: $response" >> "$REMOTE_LOG"
        fi
    fi

    # Fetch the updated queue information
    sampler_queue_info=$(curl -s --connect-timeout 10 --max-time 30 -s -u guest:guest http://rabbitmqFW:15672/api/queues | jq '.[] | select(.name == "sampler_queue")')

    # Extract the number of messages and consumers
    sampler_queue_count=$(echo "$sampler_queue_info" | jq '.messages')
    sampler_queue_consumers=$(echo "$sampler_queue_info" | jq '.consumers')

    echo "$(date) - Sampler queue size: $sampler_queue_count, Consumers: $sampler_queue_consumers" >> "$REMOTE_LOG"

    # Only execute the script if there are more than 30 messages in the queue or no consumers attached
    if (( sampler_queue_count > 30 )) || (( sampler_queue_consumers == 0 )); then
        echo "$(date) - More than 30 messages in the sampler queue. Checking GPU resources..." >> "$REMOTE_LOG"

        # Variable to track if a suitable server was found
        found_suitable_server=false

        # Iterate through remote servers to check their GPU utilization
        for SERVER in "${REMOTE_SERVERS_GPU[@]}"; do
            echo "$(date) - Checking GPU resources on $SERVER..." >> "$REMOTE_LOG"

            # Use ssh to check GPU stats on the remote server
            GPU_INFO=$(ssh -o ConnectTimeout=10 "franziska@$SERVER" "
                nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
            ")

            if [ $? -ne 0 ]; then
                echo "$(date) - Failed to query GPU information on $SERVER." >> "$REMOTE_LOG"
                continue  # Move to the next server
            fi

            echo "$(date) - GPU info on $SERVER: $GPU_INFO" >> "$REMOTE_LOG"

            # Parse the output and check for suitable GPUs
            total_memory=0
            suitable_gpus=()
            gpu_id=0

            # Parse the GPU memory and utilization values
            while IFS=',' read -r memory_free gpu_utilization; do
                if (( memory_free > GPU_MEMORY_THRESHOLD )) && (( gpu_utilization < GPU_UTILIZATION_THRESHOLD )); then
                    total_memory=$((total_memory + memory_free))
                    suitable_gpus+=($gpu_id)
                fi
                gpu_id=$((gpu_id + 1))
            done <<< "$GPU_INFO"

            echo "$(date) - Total available GPU memory on $SERVER: $total_memory MiB" >> "$REMOTE_LOG"

            if (( total_memory < TOTAL_MEMORY_REQUIRED )); then
                echo "$(date) - Not enough combined GPU memory available on $SERVER. Skipping..." >> "$REMOTE_LOG"
                continue
            fi

            # Proceed to start the script if enough memory is available
            CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${suitable_gpus[*]}")

            echo "$(date) - Starting funsearch_s.py on $SERVER with GPUs: $CUDA_VISIBLE_DEVICES" >> "$REMOTE_LOG"
            
            # Update the remote config before running funsearch_s.py
            echo "$(date) - Updating remote config on $SERVER" >> "$REMOTE_LOG"
            update_remote_configs "$GRID_PARAMS_FILE" "$SERVER"
            sleep 5

            # Start funsearch_s.py on the identified GPUs
            ssh -n -o ConnectTimeout=10 "franziska@$SERVER" "
                docker exec $CONTAINER_NAME /bin/bash -c '
                    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
                    cd $ROOT_DIRECTORY || echo Failed to change directory
                    echo Activating Conda environment on $SERVER
                    conda run -n fw2 python $SCRIPT_PATH_REMOTE_S || echo Conda environment or Python execution failed
                '
            " >> "$REMOTE_LOG" 2>&1 &

            ssh_exit_status=$?
            if [ $ssh_exit_status -eq 0 ]; then
                echo "$(date) - Successfully initiated funsearch_s.py on $SERVER." >> "$REMOTE_LOG"
                found_suitable_server=true
                break  # Exit after finding one suitable server
            else
                echo "$(date) - Failed to initiate funsearch_s.py on $SERVER. SSH command failed with exit status $ssh_exit_status." >> "$REMOTE_LOG"
                continue  # Move to the next server
            fi

            # Wait for 10 seconds and then check if the number of consumers increased
            sleep 10

            # Fetch the updated number of consumers
            updated_consumers=$(curl -s --connect-timeout 10 --max-time 30 -s -u guest:guest http://rabbitmqFW:15672/api/queues | jq '.[] | select(.name == "sampler_queue") | .consumers')

            echo "$(date) - Initial consumers: $initial_consumers, Updated consumers: $updated_consumers" >> "$REMOTE_LOG"

            if (( updated_consumers == initial_consumers + 1 )); then
                echo "$(date) - funsearch_s.py started successfully on $SERVER. Consumers increased." >> "$REMOTE_LOG"
                found_suitable_server=true
                break  # Exit after finding one suitable server
            else
                echo "$(date) - funsearch_s.py failed to start on $SERVER or consumers did not increase." >> "$REMOTE_LOG"
            fi
        done

        # If no suitable server is found after iterating over all servers
        if ! $found_suitable_server; then
            echo "$(date) - No suitable server with sufficient GPU memory found. Exiting." >> "$REMOTE_LOG"
            return  # Return early to avoid infinite loop
        fi
    else
        echo "$(date) - Sampler queue has less than 30 messages. Skipping execution." >> "$REMOTE_LOG"
    fi
}




# Function to monitor GPU utilization and kill idle container processes inside the container
check_gpu_utilization_and_kill_idle_processes() {
    echo "$(date) - Starting GPU utilization monitoring on remote servers..." >> "$REMOTE_LOG"

    # Time threshold for killing idle processes (15 minutes in seconds)
    TIME_THRESHOLD=900  # 15 minutes

    # Declare an associative array to track idle time for each process
    declare -A idle_time_tracker

    while true; do
        for SERVER in "${REMOTE_SERVERS[@]}"; do
            echo "$(date) - Checking GPU utilization on $SERVER..." >> "$REMOTE_LOG"

            # Fetch GPU memory and utilization stats using nvidia-smi on the remote server
            GPU_STATS=$(ssh -o ConnectTimeout=10 "franziska@$SERVER" "
                nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
            ")

            # Log the output of the ssh command (whether it's empty or not)
            if [ -z "$GPU_STATS" ]; then
                echo "$(date) - No output from nvidia-smi on $SERVER. GPU_STATS is empty." >> "$REMOTE_LOG"
            else
                echo "$(date) - GPU stats from $SERVER: $GPU_STATS" >> "$REMOTE_LOG"
            fi

            # Check if ssh command failed
            if [ $? -ne 0 ]; then
                echo "$(date) - Failed to query GPU utilization on $SERVER. SSH command failed." >> "$REMOTE_LOG"
                echo "$(date) - SSH output: $GPU_STATS" >> "$REMOTE_LOG"
                continue  # Move to the next server
            fi

            # Parse the output of nvidia-smi
            echo "$GPU_STATS" | while IFS=',' read -r pid gpu_uuid memory_used gpu_utilization; do
                # Check if the process belongs to the pytorchFW container
                PROCESS_IN_CONTAINER=$(ssh -o ConnectTimeout=10 "franziska@$SERVER" "
                    docker exec $CONTAINER_NAME ps -p $pid -o pid=
                ")

                if [[ -n "$PROCESS_IN_CONTAINER" ]]; then
                    # Check if GPU utilization is 0%
                    if (( gpu_utilization == 0 )); then
                        # Increment idle time if GPU utilization is 0%
                        if [[ -n "${idle_time_tracker[$pid]}" ]]; then
                            idle_time_tracker[$pid]=$(( ${idle_time_tracker[$pid]} + 300 ))  # Add 5 minutes
                        else
                            idle_time_tracker[$pid]=300  # Start tracking the process (5 minutes)
                        fi

                        echo "$(date) - Process $pid on GPU $gpu_uuid has been idle for ${idle_time_tracker[$pid]} seconds." >> "$REMOTE_LOG"

                        # Kill the process if idle time exceeds 15 minutes
                        if (( ${idle_time_tracker[$pid]} >= TIME_THRESHOLD )); then
                            echo "$(date) - Attempting to kill process $pid inside container $CONTAINER_NAME on $SERVER (idle for more than 15 minutes)." >> "$REMOTE_LOG"
                            ssh -o ConnectTimeout=10 "franziska@$SERVER" "docker exec $CONTAINER_NAME kill $pid"

                            # Check if the process is still running inside the container
                            PROCESS_CHECK=$(ssh -o ConnectTimeout=10 "franziska@$SERVER" "
                                docker exec $CONTAINER_NAME ps -p $pid
                            ")

                            if [[ "$PROCESS_CHECK" == *"$pid"* ]]; then
                                echo "$(date) - Process $pid is still running inside the container. Attempting force kill." >> "$REMOTE_LOG"
                                ssh -o ConnectTimeout=10 "franziska@$SERVER" "docker exec $CONTAINER_NAME kill -9 $pid"

                                # Check again if the process was killed
                                PROCESS_CHECK_FORCE=$(ssh -o ConnectTimeout=10 "franziska@$SERVER" "
                                    docker exec $CONTAINER_NAME ps -p $pid
                                ")

                                if [[ "$PROCESS_CHECK_FORCE" == *"$pid"* ]]; then
                                    echo "$(date) - Force kill failed for process $pid inside container on $SERVER. Restarting the pytorchFW container." >> "$REMOTE_LOG"
                                    ssh -o ConnectTimeout=10 "franziska@$SERVER" "docker restart $CONTAINER_NAME"
                                else
                                    echo "$(date) - Force kill succeeded for process $pid inside the container on $SERVER." >> "$REMOTE_LOG"
                                fi
                            else
                                echo "$(date) - Process $pid successfully killed inside the container on $SERVER." >> "$REMOTE_LOG"
                            fi

                            unset idle_time_tracker[$pid]  # Remove the process from the tracker
                        fi
                    else
                        # Reset idle time if the process starts using the GPU again
                        unset idle_time_tracker[$pid]
                    fi
                fi
            done
        done

        # Sleep for 5 minutes before checking again
        sleep 300
    done
}

# Main script execution loop
while true; do
    # Run the commands concurrently
    kill_python_processes &
    restart_remote_containers 
    wait 

    check_and_delete_rabbitmq_resources 

    # Wait for all background tasks to complete
    sleep 10    

    echo "$(date) - Starting grid search..." >> "$REMOTE_LOG"
    start_time=$(date +%s)  # Capture the start time of the loop

    export CUDA_VISIBLE_DEVICES=0,1

    # Start the local grid search script in the background
    python /franziska/implementation/grid_search.py >> "$REMOTE_LOG" 2>&1 &

    # Start the remote scripts after the grid_search.py has started
    for SERVER in "${REMOTE_SERVERS[@]}"; do
        echo "Starting Evaluator script on $SERVER using Docker container $CONTAINER_NAME" >> "$REMOTE_LOG"
        ssh -n -o ConnectTimeout=10 "franziska@$SERVER" "
            docker exec $CONTAINER_NAME /bin/bash -c '
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                cd $ROOT_DIRECTORY || echo Failed to change directory
                echo Activating Conda environment on $SERVER
                conda run -n fw2 python $SCRIPT_PATH_REMOTE_E || echo Conda environment or Python execution failed # run funsearch_e.py
            '
        " >> "$REMOTE_LOG" 2>&1 &
    done
    sleep 30
    # Timeout for periodic queue check every 5 minutes
    while true; do
        current_time=$(date +%s)
        elapsed_time=$(( current_time - start_time ))

        if (( elapsed_time >= TIMEOUT_DURATION )); then
            echo "$(date) - Timeout reached. Proceeding to terminate and restart processes." >> "$REMOTE_LOG"
            break  # Exit the while loop when timeout is reached
        fi

        # Check the sampler queue and run the script if suitable
        check_sampler_queue_and_execute

        # Sleep for 300 seconds (5 minutes) between checks
        sleep 600
    done

    # Timeout reached, terminate all processes
    echo "$(date) - Timeout reached. Terminating processes..." >> "$REMOTE_LOG"
done
