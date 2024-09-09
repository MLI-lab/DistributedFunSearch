import os
import time
import pickle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Path to the checkpoint file and output HTML file
CHECKPOINT_FILEPATH = "/franziska/implementation/checkpoint.pkl"
HTML_OUTPUT_PATH = "/franziska/implementation/performance_over_time.html"

# Initialize the Plotly figure with 4 rows and 3 columns
fig = make_subplots(rows=4, cols=3, shared_xaxes=False, vertical_spacing=0.1,
                    subplot_titles=['Performance Over Time'] + [f'Cluster Diversity for Island {i+1}' for i in range(10)])

# Configure the layout for the entire figure
fig.update_layout(
    title='Island and Cluster Analysis',
    legend_title='Islands',
    hovermode='closest'
)

# Function to update the cluster diversity plot for each island
def update_cluster_visualization(selected_island, islands_state, last_reset_time, row, col):
    island = islands_state[selected_island]
    clusters = island.get('clusters', {})

    # Convert to numpy.datetime64 for better handling in Plotly
    time_value = np.datetime64(datetime.fromtimestamp(last_reset_time))

    x_values = []
    y_values = []
    sizes = []
    labels = []

    # Increased jitter magnitude and applied jitter directly to datetime
    jitter = np.random.uniform(-5, 5, size=len(clusters))  # Jitter in minutes

    for i, (cluster_signature, cluster_info) in enumerate(clusters.items()):
        cluster_score = cluster_info.get('score', 0)
        cluster_size = len(cluster_info.get('programs', []))

        # Apply jitter in minutes
        jitter_timedelta = np.timedelta64(int(jitter[i] * 60), 's')
        x_values.append(time_value + jitter_timedelta)
        y_values.append(cluster_score)
        sizes.append(cluster_size * 20)
        labels.append(f"Cluster Score: {cluster_score}, Size: {cluster_size}")

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(size=sizes, color='rgba(100, 200, 255, 0.6)', sizemode='area', sizeref=2.0 * max(sizes) / (40. ** 2), sizemin=4),
            text=labels,
            name=f"Island {selected_island + 1}"
        ),
        row=row, col=col
    )

# Function to update the entire visualization and pre-load all islands' data
def update_visualization():
    if os.path.exists(CHECKPOINT_FILEPATH):
        with open(CHECKPOINT_FILEPATH, "rb") as f:
            checkpoint_data = pickle.load(f)

        best_scores_per_island = checkpoint_data.get("best_score_per_island", [])
        islands_state = checkpoint_data.get("islands_state", [])
        
        # Start with the current time
        last_reset_time = time.time()

        # Update performance over time for all islands
        for idx, score in enumerate(best_scores_per_island):
            # Increment the time to simulate a time step
            current_time = last_reset_time + idx * 60  # Increment by 60 seconds (for example)
            fig.add_trace(
                go.Scatter(x=[datetime.fromtimestamp(current_time)], y=[score], mode='lines+markers', name=f'Island {idx + 1} Performance'),
                row=1, col=1
            )

        # Update cluster diversity data for each island
        row, col = 1, 2
        for idx in range(len(islands_state)):
            if col > 3:
                row += 1
                col = 1
            
            # Use a different time value for each island's cluster update
            update_cluster_visualization(idx, islands_state, last_reset_time + idx * 60, row, col)
            col += 1

        fig.write_html(HTML_OUTPUT_PATH, auto_open=False)

    else:
        print("Checkpoint file not found.")

# Function to check file modification and update
def check_for_updates(filepath, interval=60):
    last_mtime = None
    while True:
        try:
            current_mtime = os.path.getmtime(filepath)
            if last_mtime is None:
                last_mtime = current_mtime
            elif current_mtime > last_mtime:
                print("File has been modified. Updating visualization...")
                last_mtime = current_mtime
                update_visualization()
        except FileNotFoundError:
            print("File not found. Waiting for it to appear...")
        except Exception as e:
            print("Error checking file:", e)
        time.sleep(interval)

# Main execution block
if __name__ == "__main__":
    print("Monitoring changes to:", CHECKPOINT_FILEPATH)
    check_for_updates(CHECKPOINT_FILEPATH)
