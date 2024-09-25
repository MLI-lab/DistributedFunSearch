#!/bin/bash

# List of remote servers
REMOTE_SERVERS=("sequoia.mli.ei.tum.de" "gpumlp2.msv.ei.tum.de" "zion.msv.ei.tum.de" "bigsur.mli.ei.tum.de")

# Docker container name
CONTAINER_NAME="pytorchFW"

# Paths to the Python scripts
SCRIPT_PATH_REMOTE="/franziska/Funsearch/implementation/funsearch_e.py"
SCRIPT_PATH2_REMOTE="/franziska/Funsearch/implementation/funsearch_s.py"
ROOT_DIRECTORY="/franziska/Funsearch/implementation/"

# Timeout duration in seconds
TIMEOUT_DURATION=300  

# Log file for remote output
REMOTE_LOG="remote.log"

# Path to current grid search parameters
GRID_PARAMS_FILE="/path/to/GridSearch/current_grid_params.json"

# Minimum memory requirement (30 GB in MB)
MINIMUM_MEMORY_REQUIRED=30000  # 30 GB

# Function to get available GPUs with sufficient free memory and low utilization
get_available_gpus() {
    local server=$1
    local free_gpus=()
    local total_free_memory=0

    # Query GPU memory and utilization using nvidia-smi
    gpu_info=$(ssh "franziska@$server" "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits")

    # Iterate through the GPU info
    while IFS=',' read -r gpu_id free_mem gpu_util; do
        free_mem=$(echo "$free_mem" | xargs)  # Trim any spaces
        gpu_util=$(echo "$gpu_util" | xargs)  # Trim any spaces

        # Check if the GPU has at least 10 GB free and utilization is below 10%
        if [ "$free_mem" -ge 10000 ] && [ "$gpu_util" -le 10 ]; then
            free_gpus+=("$gpu_id")
            total_free_memory=$((total_free_memory + free_mem))
        fi
    done <<< "$gpu_info"

    # Check if the combined free memory is at least 30 GB
    if [ "$total_free_memory" -ge "$MINIMUM_MEMORY_REQUIRED" ]; then
        echo "${free_gpus[*]}"  # Return the IDs of the free GPUs
    else
        echo ""  # Return empty if no sufficient memory or utilization
    fi
}

# Function to kill all Python processes on the main server
kill_python_processes() {
    echo "$(date) - Killing all Python processes on the main server..." >> "$REMOTE_LOG"
    
    if pkill -f python; then
        echo "$(date) - All Python processes on the main server killed successfully." >> "$REMOTE_LOG"
    else
        echo "$(date) - Failed to kill Python processes on the main server." >> "$REMOTE_LOG"
    fi
}

# Function to kill all Python processes on remote servers
kill_remote_python_processes() {
    echo "$(date) - Stopping remote Python scripts..." >> "$REMOTE_LOG"
    
    for SERVER in "${REMOTE_SERVERS[@]}"; do
        echo "$(date) - Attempting to stop Python processes on $SERVER..." >> "$REMOTE_LOG"
        
        # SSH into the server and kill the Python processes in Docker container
        ssh -o ConnectTimeout=10 "franziska@$SERVER" "
            if docker exec $CONTAINER_NAME pkill -f python; then
                echo '$(date) - Python processes on $SERVER stopped successfully.' >> /remote_script.log
            else
                echo '$(date) - Failed to stop Python processes on $SERVER.' >> /remote_script.log
            fi
        " >> "$REMOTE_LOG" 2>&1 &
    done
    
    # Wait for all background tasks (killing processes on remote servers) to finish
    wait
    
    echo "$(date) - Remote Python scripts stopped." >> "$REMOTE_LOG"
}

# Function to extract parameter names and values from the JSON file and update the remote config
update_remote_configs() {
    local config_file="$1"

    # Get the list of parameter names from the JSON file
    param_names=$(jq -r 'keys[]' "$config_file")

    # Loop through the parameter names and update each one on the remote servers
    for param_name in $param_names; do
        param_value=$(jq -r --arg key "$param_name" '.[$key]' "$config_file")

        # Loop through all remote servers and update the parameter
        for SERVER in "${REMOTE_SERVERS[@]}"; do
            update_remote_config_param "$SERVER" "$param_name" "$param_value"
        done
    done
}

# Function to update a single parameter in the remote config file
update_remote_config_param() {
    local server=$1
    local param_name=$2
    local param_value=$3

    echo "$(date) - Updating $param_name=$param_value on $server" >> "$REMOTE_LOG"
    
    # Update the specific parameter in the remote config file using jq
    ssh "franziska@$server" "
        jq --arg value \"$param_value\" '.[$param_name]=$value' /franziska/Funsearch/implementation/remote_config.json > /franziska/Funsearch/implementation/temp_config.json &&
        mv /franziska/Funsearch/implementation/temp_config.json /franziska/Funsearch/implementation/remote_config.json
    " --arg param_name "$param_name"
}

# Function to check and delete RabbitMQ resources
check_and_delete_rabbitmq_resources() {
    echo "$(date) - Checking for leftover RabbitMQ resources..."

    # Fetch and delete all queues
    echo "Fetching queues..."
    queues_json=$(curl -s -u guest:guest http://rabbitmqFW:15672/api/queues)
    echo $queues_json | jq -c '.[] | .name' | while read queue; do
        queue=$(echo $queue | sed 's/"//g')  # Remove quotes from queue name
        echo "Deleting queue: $queue"
        url="http://rabbitmqFW:15672/api/queues/%2F/$queue"
        curl -X DELETE -u guest:guest "$url"
        echo "Deleted queue: $queue"
    done

    # Fetch and close all connections
    echo "Fetching connections..."
    connections_json=$(curl -s -u guest:guest http://rabbitmqFW:15672/api/connections)
    echo $connections_json | jq -c '.[] | .name' | while read connection; do
        connection=$(echo $connection | sed 's/"//g' | sed 's/->/%2D%3E/g' | sed 's/ /%20/g')  # Remove quotes, encode '->', and spaces
        url="http://rabbitmqFW:15672/api/connections/$connection"
        curl -X DELETE -u guest:guest "$url"
        echo "Closed connection: $connection"
    done

    # Fetch and close all channels
    echo "Fetching channels..."
    channels_json=$(curl -s -u guest:guest http://rabbitmqFW:15672/api/channels)
    echo $channels_json | jq -c '.[] | .name' | while read channel; do
        channel=$(echo $channel | sed 's/"//g' | sed 's/->/%2D%3E/g' | sed 's/ /%20/g')  # Remove quotes, encode '->', and spaces
        url="http://rabbitmqFW:15672/api/channels/$channel"
        curl -X DELETE -u guest:guest "$url"
        echo "Closed channel: $channel"
    done

    echo "$(date) - RabbitMQ resources cleaned up."
}

<<<<<<< HEAD
# Main script execution loop
while true; do
    echo "$(date) - Starting grid search..."

    # Set the CUDA_VISIBLE_DEVICES environment variable to limit GPU visibility
    export CUDA_VISIBLE_DEVICES=2

    # Start the local grid search script
    nohup python /franziska/implementation/grid_search.py >> "$REMOTE_LOG" 2>&1 &
    GRID_SEARCH_PID=$!

    # Update the remote configuration files with the latest parameters
    update_remote_configs "$GRID_PARAMS_FILE"

    # Start remote scripts in Docker containers
    for SERVER in "${REMOTE_SERVERS[@]}"; do
        # Check for available GPUs with at least 30 GB combined memory and low utilization
        free_gpus=$(get_available_gpus "$SERVER")

        if [ -n "$free_gpus" ]; then
            # GPUs are available with sufficient memory and low utilization
            echo "$(date) - Found free GPUs ($free_gpus) with sufficient memory on $SERVER" >> "$REMOTE_LOG"
            
            # Log and execute commands inside the Docker container on the remote server
            ssh "franziska@$SERVER" "
                docker exec $CONTAINER_NAME /bin/bash -c '
                    export CUDA_VISIBLE_DEVICES=$free_gpus  # Set GPU visibility inside the container
                    cd $ROOT_DIRECTORY || echo Failed to change directory >> /remote_script.log

                    # Log the activation of the conda environment
                    echo Activating Conda environment on $SERVER >> /remote_script.log

                    # Run funsearch_e.py script
                    conda run -n fw2 python $SCRIPT_PATH_REMOTE || echo Conda environment or Python execution failed on $SERVER >> /remote_script.log

                    # Run funsearch_s.py after funsearch_e.py
                    conda run -n fw2 python $SCRIPT_PATH2_REMOTE || echo Conda environment or Python execution failed for funsearch_s.py on $SERVER >> /remote_script.log
                '
            " >> "$REMOTE_LOG" 2>&1 &
        else
            echo "$(date) - No sufficient free GPU memory or utilization on $SERVER, skipping..." >> "$REMOTE_LOG"
        fi
    done

    # Wait for the timeout duration
    sleep "$TIMEOUT_DURATION"

    # Timeout reached, terminate all processes
    echo "$(date) - Timeout reached. Terminating processes..." >> "$REMOTE_LOG"
    kill_remote_python_processes &
    kill_python_processes &
    wait

    # Wait a few seconds for processes to terminate
    sleep 5

    # Check and delete RabbitMQ resources
    check_and_delete_rabbitmq_resources

    echo "$(date) - Grid search restarted." >> "$REMOTE_LOG"
=======
# Function to restart grid search
restart_grid_search() {
    echo "$(date) - Restarting grid search..."
    export CUDA_VISIBLE_DEVICES=2,3  # Export environment variable
    timeout 3690 python /franziska/implementation/grid_search.py
    echo "$(date) - Grid search script terminated. Restarting..."
}

# Main script execution loop
while true; do
    kill_python_processes
    sleep 5
    check_and_delete_rabbitmq_resources
    sleep 10
    restart_grid_search
>>>>>>> 0516afe2cd719dcd52224bde74ea56fcd62f16df
done
