import os
import time
import pickle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import math


# Set CHECKPOINT_DIR to the current directory
CHECKPOINT_DIR = os.getcwd()
HTML_OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "performance_over_time.html")

# Figure setup
fig = make_subplots(rows=4, cols=3, shared_xaxes=False, vertical_spacing=0.1,
                    subplot_titles=['Performance Over Time'] + [f'Cluster Diversity for Island {i+1}' for i in range(10)])

# Adjust title and font sizes globally
fig.update_layout(
    title='Island and Cluster Analysis',
    legend_title='Islands',
    hovermode='closest',
    font=dict(size=10),  # Smaller font size for general text
)

# Apply specific settings to x-axes to reduce the label font size
for i in range(1, 12):
    fig.update_xaxes(row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1, tickfont=dict(size=8))  # Make x-axis labels smaller

# Store island performance data globally
island_performance_data = {}

# Update performance plot with lines connecting points for each island
def update_performance_plot(checkpoint_data, file_mtime):
    time_value = datetime.fromtimestamp(file_mtime)  # Convert file modification time to datetime
    
    for idx, score in enumerate(checkpoint_data.get("best_score_per_island", [])):
        island_id = idx + 1
        
        # If the island doesn't exist in the performance data, initialize it
        if island_id not in island_performance_data:
            island_performance_data[island_id] = {'times': [], 'scores': []}
        
        # Append the current time and score to the respective island's data
        island_performance_data[island_id]['times'].append(time_value)
        island_performance_data[island_id]['scores'].append(score)
        
        # Plot the island's performance over time (connecting the dots with lines)
        fig.add_trace(
            go.Scatter(
                x=island_performance_data[island_id]['times'], 
                y=island_performance_data[island_id]['scores'], 
                mode='lines+markers', 
                name=f'Island {island_id} Performance',
                marker=dict(size=8),  # Marker size can be adjusted here
                line=dict(width=2)    # Line width can also be adjusted here
            ),
            row=1, col=1
        )

def update_cluster_visualization(selected_island, islands_state, file_mtime, row, col):
    island = islands_state[selected_island]
    clusters = island.get('clusters', {})
    time_value = np.datetime64(datetime.fromtimestamp(file_mtime))
    jitter = np.random.uniform(-5, 5, size=len(clusters))  # Jitter in minutes

    sizes = []  # Define sizes array to store the size of each cluster

    # Initialize loop index
    i = 0

    # Calculate marker sizes for clusters with logarithmic scaling
    for cluster_key, cluster_info in clusters.items():
        cluster_size = len(cluster_info.get('programs', []))  # Get the size of the cluster (number of programs)
        log_size = math.log(cluster_size + 1)  # Apply logarithmic scaling to avoid size 0 for empty clusters
        scaled_size = log_size * 10  # Adjust scaling factor for marker size
        sizes.append(scaled_size)  # Append the calculated size

        jitter_timedelta = np.timedelta64(int(jitter[i] * 60), 's')  # Apply jitter for visualization
        jittered_time = time_value + jitter_timedelta  # Jittered time

        # Add the true time to the hover text
        fig.add_trace(
            go.Scatter(
                x=[jittered_time],
                y=[cluster_info.get('score', 0)],
                mode='markers',
                marker=dict(size=sizes[i], color='rgba(100, 200, 255, 0.6)', sizemode='area', sizeref=2.0 * max(sizes) / (40. ** 2), sizemin=4),
                text=f"True Time: {datetime.fromtimestamp(file_mtime)}<br>Signature: {cluster_key}<br>Size: {cluster_size}",
                name=f"Island {selected_island + 1}"
            ),
            row=row, col=col
        )

        # Increment size index in the loop
        i += 1



# Update visualization
def update_visualization(checkpoint_filepath, file_mtime):
    if os.path.exists(checkpoint_filepath):
        with open(checkpoint_filepath, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        # Update performance plot (with lines connecting points)
        update_performance_plot(checkpoint_data, file_mtime)
        
        # Update cluster plots with jitter
        row, col = 1, 2
        for idx in range(len(checkpoint_data.get("islands_state", []))):
            if col > 3:
                row += 1
                col = 1
            update_cluster_visualization(idx, checkpoint_data['islands_state'], file_mtime + idx * 60, row, col)
            col += 1
        
        fig.write_html(HTML_OUTPUT_PATH, auto_open=False)
    else:
        print("Checkpoint file not found.")

# Monitor for new checkpoint files
def check_for_new_checkpoints(directory, interval=60):
    last_seen_timestamp = None

    while True:
        try:
            # List all checkpoint files
            checkpoint_files = [f for f in os.listdir(directory) if f.startswith("checkpoint_") and f.endswith(".pkl")]
            
            checkpoint_timestamps = []
            for f in checkpoint_files:
                # Split the filename to extract the date-time part
                parts = f.split("_")
                if len(parts) > 2:
                    # Join the date and time parts together, ignoring the 'pkl' extension
                    timestamp_str = "_".join(parts[1:3])
                    # Remove the '.pkl' from the time part
                    timestamp_str = timestamp_str.replace('.pkl', '')
                    # Convert the date-time string to a datetime object
                    timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                    # Convert the datetime object to a Unix timestamp
                    timestamp = timestamp_dt.timestamp()
                    checkpoint_timestamps.append((timestamp, f))
                else:
                    print(f"Filename does not have the correct format to extract timestamp: {f}")

            if checkpoint_timestamps:
                # Find the most recent checkpoint file
                latest_timestamp, latest_file = max(checkpoint_timestamps, key=lambda x: x[0])
                
                if last_seen_timestamp is None or latest_timestamp > last_seen_timestamp:
                    print(f"New checkpoint detected: {latest_file}")
                    # Update visualization
                    checkpoint_filepath = os.path.join(directory, latest_file)
                    update_visualization(checkpoint_filepath, latest_timestamp)
                    last_seen_timestamp = latest_timestamp

        except Exception as e:
            print(f"Error while checking for new checkpoints: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    print(f"Monitoring new checkpoints in directory: {CHECKPOINT_DIR}")
    check_for_new_checkpoints(CHECKPOINT_DIR)

