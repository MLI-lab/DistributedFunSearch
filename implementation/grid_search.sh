#!/bin/bash

# List of remote servers
#REMOTE_SERVERS=("sequoia.mli.ei.tum.de" "gpumlp2.msv.ei.tum.de" "gpumlp.msv.ei.tum.de" "zion.msv.ei.tum.de" "bigsur.mli.ei.tum.de")
REMOTE_SERVERS=("zion.msv.ei.tum.de" )

# Docker container name
CONTAINER_NAME="pytorchFW"

# Path to the Python script
SCRIPT_PATH_REMOTE="/franziska/Funsearch/implementation/funsearch_e.py"
ROOT_DIRECTORY="/franziska/Funsearch/implementation/"

# Timeout duration in seconds
TIMEOUT_DURATION=3630  

# Log file for remote output
REMOTE_LOG="remote.log"

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
        ssh -o ConnectTimeout=10 "franziska@$SERVER" "
            docker restart $CONTAINER_NAME
        " >> "$REMOTE_LOG" 2>&1
        if [ $? -eq 0 ]; then
            echo "$(date) - Docker container on $SERVER restarted successfully." >> "$REMOTE_LOG"
        else
            echo "$(date) - Failed to restart Docker container on $SERVER." >> "$REMOTE_LOG"
        fi
    done    
    echo "$(date) - Remote Docker containers restarted." >> "$REMOTE_LOG"
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

    # Fetch and close all connections
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

    # Final confirmation message
    echo "$(date) - RabbitMQ resources cleaned up." >> "$REMOTE_LOG"
}

# Main script execution loop
while true; do
    restart_remote_containers
    kill_python_processes
    check_and_delete_rabbitmq_resources
    sleep 10
    echo "$(date) - Starting grid search..." >> "$REMOTE_LOG"
    
    # Set the CUDA_VISIBLE_DEVICES environment variable to limit GPU visibility
    export CUDA_VISIBLE_DEVICES=0,3

    # Start the local grid search script in the background
    python /franziska/implementation/grid_search.py >> "$REMOTE_LOG" 2>&1 &
    
    # Start the remote scripts after the grid_search.py has started
    for SERVER in "${REMOTE_SERVERS[@]}"; do
        echo "Starting Python script on $SERVER using Docker container $CONTAINER_NAME" >> "$REMOTE_LOG"
        ssh -o ConnectTimeout=10 "franziska@$SERVER" "
            docker exec $CONTAINER_NAME /bin/bash -c '
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                cd $ROOT_DIRECTORY || echo Failed to change directory
                echo Activating Conda environment on $SERVER
                conda run -n fw2 python $SCRIPT_PATH_REMOTE || echo Conda environment or Python execution failed
            '
        " >> "$REMOTE_LOG" 2>&1 &
    done

    # Wait for the timeout duration without blocking
    echo "$(date) - Sleeping for $TIMEOUT_DURATION seconds..." >> "$REMOTE_LOG"
    sleep $TIMEOUT_DURATION 
    
    # Timeout reached, terminate all processes
    echo "$(date) - Timeout or process end reached. Terminating processes..." >> "$REMOTE_LOG"
    restart_remote_containers
    sleep 5
    kill_python_processes
    sleep 10

    # Check and delete RabbitMQ resources
    check_and_delete_rabbitmq_resources
    sleep 20 
    echo "$(date) - Grid search restarted." >> "$REMOTE_LOG"
done
