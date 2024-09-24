#!/bin/bash

# Function to kill all Python processes
kill_python_processes() {
    echo "$(date) - Killing all Python processes..."
    pkill -f python
    echo "$(date) - All Python processes killed."
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
done
