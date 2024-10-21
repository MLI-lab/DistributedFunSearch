#!/bin/bash

# List of remote servers
REMOTE_SERVERS=("sequoia.mli.ei.tum.de" "gpumlp.msv.ei.tum.de" "zion.msv.ei.tum.de" "gpumlp2.msv.ei.tum.de")

# Corresponding memory limits and CPU limits for each server
memory_limits=("120G" "75G" "75G" "75G")
cpu_limits=("110" "55" "55" "55")

# Function to check GPU utilization and memory on local/remote system
check_gpus() {
    # Use nvidia-smi to get utilization and memory info
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
    CUDA_VISIBLE_DEVICES=""

    # Iterate through each GPU and check its utilization and memory
    while IFS=, read -r gpu_id free_memory gpu_utilization; do
        if [[ $free_memory -gt 32768 && $gpu_utilization -lt 50 ]]; then
            if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
                CUDA_VISIBLE_DEVICES="$gpu_id"
            else
                CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$gpu_id"
            fi
        fi
    done <<< "$GPU_INFO"

    echo "$CUDA_VISIBLE_DEVICES"
}

# Check local GPUs and set CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(check_gpus)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No suitable GPUs found on the local machine."
    exit 1
fi

# Activate Conda environment and run Python script locally with the selected GPUs
echo "Activating Conda environment and running funsearch.py locally"
source /opt/conda/etc/profile.d/conda.sh  # Ensure the Conda environment can be activated
conda activate fw2
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python /franziska/implementation/funsearch.py &
echo "funsearch.py started locally."

# Function to configure and run scripts on a remote server
run_remote_container() {
    local server=$1
    local memory_limit=$2
    local cpu_limit=$3

    echo "Starting on remote server: $server"

    ssh "franziska@$server" << EOF
        # Update the memory and CPU limits in the Docker configuration separately
        docker update --memory $memory_limit pytorchFW
        docker update --cpus $cpu_limit pytorchFW

        # Check GPUs again on the remote server
        GPU_INFO=\$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
        CUDA_VISIBLE_DEVICES=""
        while IFS=, read -r gpu_id free_memory gpu_utilization; do
            if [[ \$free_memory -gt 20480 && \$gpu_utilization -lt 50 ]]; then
                if [ -z "\$CUDA_VISIBLE_DEVICES" ]; then
                    CUDA_VISIBLE_DEVICES="\$gpu_id"
                else
                    CUDA_VISIBLE_DEVICES="\$CUDA_VISIBLE_DEVICES,\$gpu_id"
                fi
            fi
        done <<< "\$GPU_INFO"

        echo "CUDA_VISIBLE_DEVICES found on $server: \$CUDA_VISIBLE_DEVICES"

        # Enter the container and run the scripts
        docker exec pytorchFW bash -c "
            source /opt/conda/etc/profile.d/conda.sh
            conda activate fw2
            cd /franziska/Funsearch/implementation/
            CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python funsearch_s.py &
            echo 'funsearch_s.py started on $server.'
            python funsearch_e.py
            echo 'funsearch_e.py started on $server.'
        "
EOF
}

# Loop through each remote server and run the scripts with the corresponding memory and CPU limits
for i in "${!REMOTE_SERVERS[@]}"; do
    run_remote_container "${REMOTE_SERVERS[$i]}" "${memory_limits[$i]}" "${cpu_limits[$i]}"
done

echo "All remote processes started."
