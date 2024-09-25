#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/franziska/Funsearch/implementation/funsearch_e.py"

# Function to start the script
start_script() {
    echo "Starting the script..."
    python3 "$SCRIPT_PATH" &
}

# Function to stop the script
stop_script() {
    # Find the process ID of the running script
    PID=$(ps aux | grep "$SCRIPT_PATH" | grep -v grep | awk '{print $2}')

    # Check if the process is running
    if [ ! -z "$PID" ]; then
        echo "Stopping script with PID: $PID"
        kill -9 "$PID"  # Forcefully stop the script
    else
        echo "Script is not running."
    fi
}

while true; do
    # Start the script
    start_script

    # Wait for 3600 seconds (1 hour) before stopping the script
    sleep 3600

    # Stop the script
    stop_script

    # Sleep for 100 seconds before repeating
    echo "Sleeping for 100 seconds..."
    sleep 100
done
