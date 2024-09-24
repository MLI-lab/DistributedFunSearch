import os
import pickle
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# Import your similarity function
from similarity import compare_one_code_similarity_with_protection

# Set CHECKPOINT_DIR to your 'Checkpoints' folder
CHECKPOINT_DIR = os.path.join(os.getcwd(), 'Checkpoints')

# Function to calculate the similarity matrix using your similarity function
def compute_similarity_matrix(programs, similarity_type, protected_vars):
    print("compute_similarity_matrix...")
    n = len(programs)
    similarity_matrix = np.zeros((n, n))
    if n == 1:
        return np.array([[1]])  # Edge case: only one program
    for i in range(n):
        for j in range(i, n):  # Only compute upper triangle and diagonal
            similarity = compare_one_code_similarity_with_protection(
                programs[i], programs[j], similarity_type, protected_vars
            )
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix
    return similarity_matrix

# Function to compute similarity between clusters within an island
def compute_cluster_similarity(island, similarity_type, protected_vars):
    print("compute_cluster_similarity...")
    clusters = list(island.get('clusters', {}).values())
    n_clusters = len(clusters)
    
    # Initialize similarity matrix for clusters
    similarity_matrix = np.zeros((n_clusters, n_clusters))

    # Iterate over each pair of clusters and compute similarity between their most similar programs
    for i, cluster_a in enumerate(clusters):
        for j, cluster_b in enumerate(clusters):
            if i <= j:  # Compute only for upper triangle and diagonal
                programs_a = cluster_a.get('programs', [])
                programs_b = cluster_b.get('programs', [])
                if programs_a and programs_b:
                    # Find the most similar programs between Cluster A and Cluster B
                    max_similarity = 0
                    for prog_a in programs_a:
                        for prog_b in programs_b:
                            similarity = compare_one_code_similarity_with_protection(
                                prog_a, prog_b, similarity_type, protected_vars
                            )
                            max_similarity = max(max_similarity, similarity)
                    similarity_matrix[i, j] = max_similarity
                    similarity_matrix[j, i] = max_similarity  # Symmetric matrix
                else:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0  # No similarity if empty clusters

    return similarity_matrix

# Function to compute similarity between islands
def compute_similarity_between_islands(island_a, island_b, similarity_type, protected_vars):
    print("compute_similarity_between_islands...")
    programs_a = []
    programs_b = []

    # Collect all programs from island A and island B
    for cluster_a in island_a.get('clusters', {}).values():
        programs_a.extend(cluster_a.get('programs', []))

    for cluster_b in island_b.get('clusters', {}).values():
        programs_b.extend(cluster_b.get('programs', []))

    # Initialize similarity matrix of size (len(programs_a), len(programs_b))
    similarity_matrix = np.zeros((len(programs_a), len(programs_b)))

    # Compute similarity between each program in island A and each program in island B
    for i, prog_a in enumerate(programs_a):
        for j, prog_b in enumerate(programs_b):
            similarity_matrix[i, j] = compare_one_code_similarity_with_protection(
                prog_a, prog_b, similarity_type, protected_vars
            )

    # Sort by similarity, select top 100 most similar programs if more than 100 programs exist
    if len(programs_a) > 100 or len(programs_b) > 100:
        # Flatten the similarity matrix, find top 100 similarities
        top_similarities = np.argsort(similarity_matrix.ravel())[-100:]  # Get the indices of top 100 similarities
        top_indices_a = top_similarities // len(programs_b)
        top_indices_b = top_similarities % len(programs_b)
        programs_a = [programs_a[i] for i in top_indices_a]
        programs_b = [programs_b[i] for i in top_indices_b]
        similarity_matrix = similarity_matrix[np.ix_(top_indices_a, top_indices_b)]
    
    return similarity_matrix, programs_a, programs_b  # Return the matrix and the program lists

# Function to load all checkpoint data
def load_checkpoint_data():
    print("Loading checkpoint data...")
    files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("checkpoint_") and f.endswith(".pkl")
    ]
    island_data_list = []
    for f in files:
        parts = f.split("_")
        timestamp_str = "_".join(parts[1:3]).replace('.pkl', '')
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        with open(os.path.join(CHECKPOINT_DIR, f), "rb") as file:
            data = pickle.load(file)
            islands_state = data.get('islands_state', [])
            for idx, island in enumerate(islands_state):
                island_entry = {
                    'timestamp': timestamp_dt,
                    'island_index': idx + 1,  # Island index starting from 1
                    'island_data': island
                }
                island_data_list.append(island_entry)
    # Sort data by timestamp
    island_data_list.sort(key=lambda x: x['timestamp'])
    return island_data_list

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Cluster and Island Similarity"

# App layout
app.layout = html.Div([
    html.H1("Cluster and Island Program Similarity"),

    # Radio buttons to select comparison type
    html.Label("Select Comparison Type:"),
    dcc.RadioItems(
        id='comparison-type',
        options=[
            {'label': 'Within-Cluster Similarity', 'value': 'within_cluster'},
            {'label': 'Between-Cluster Similarity', 'value': 'between_cluster'},
            {'label': 'Between-Island Similarity', 'value': 'between_island'}
        ],
        value='within_cluster',  # Default value
        inline=True
    ),
    
    # Dropdown for selecting island
    html.Label("Select Island to Compare:"),
    dcc.Dropdown(
        id='island-dropdown',
        options=[],  # Options populated dynamically
        value=None
    ),

    # Hidden div to store the island data
    dcc.Store(id='stored-island-data', data=[]),

    # Div for holding the heatmaps
    html.Div(id='heatmap-container'),

    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 60 seconds
        n_intervals=0
    )
])

# Callback to update dropdown options and default island
@app.callback(
    [Output('island-dropdown', 'options'),
     Output('island-dropdown', 'value'),
     Output('stored-island-data', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('island-dropdown', 'value')]
)
def update_dropdown_options(n_intervals, selected_island):
    try:
        # Load data
        island_data_list = load_checkpoint_data()

        # Get unique islands
        unique_islands = sorted(set([entry['island_index'] for entry in island_data_list]))

        # Set default selection if None
        if selected_island not in unique_islands:
            selected_island = unique_islands[0] if unique_islands else None

        # Update dropdown options
        island_options = [{'label': f'Island {i}', 'value': i} for i in unique_islands]

        # Store island data in dcc.Store (convert to JSON serializable format)
        stored_data = [{'island_index': entry['island_index'], 'timestamp': entry['timestamp'].isoformat()} for entry in island_data_list]

        return island_options, selected_island, stored_data
    except Exception as e:
        print(f"Exception in update_dropdown_options: {e}")
        return [], None, []

# Callback to update heatmaps based on comparison type
@app.callback(
    Output('heatmap-container', 'children'),
    [Input('comparison-type', 'value'),
     Input('island-dropdown', 'value'),
     Input('interval-component', 'n_intervals')],
    [State('stored-island-data', 'data')]
)
def update_heatmaps(comparison_type, selected_island, n_intervals, stored_island_data):
    try:
        if selected_island is None or not stored_island_data:
            return html.Div("No data available.")

        similarity_type = 'bag_of_nodes'  # Specify your similarity type
        protected_vars = ['node', 'G', 'n', 's']  # Default protected vars

        # Reconstruct island_data_list (this is a simplified example)
        island_data_list = load_checkpoint_data()  # In a real app, avoid reloading data like this

        # Get the selected island's data
        selected_island_data = None
        for entry in island_data_list:
            if entry['island_index'] == selected_island:
                selected_island_data = entry['island_data']
                break

        if not selected_island_data:
            return html.Div("Selected island data not found.")

        # Determine which type of heatmap to display
        if comparison_type == 'within_cluster':
            return update_within_cluster_heatmaps(selected_island_data, similarity_type, protected_vars)
        elif comparison_type == 'between_cluster':
            return update_between_cluster_heatmaps(selected_island_data, similarity_type, protected_vars)
        elif comparison_type == 'between_island':
            return update_between_island_heatmaps(selected_island_data, similarity_type, protected_vars, island_data_list, selected_island)
    except Exception as e:
        print(f"Exception in update_heatmaps: {e}")
        return html.Div("An error occurred.")

# Function to generate within-cluster similarity heatmap
def update_within_cluster_heatmaps(island, similarity_type, protected_vars):
    cluster_similarity_matrix = compute_cluster_similarity(island, similarity_type, protected_vars)
    cluster_labels = [f'Cluster {i+1}' for i in range(len(island.get('clusters', {})))]

    fig = go.Figure(data=go.Heatmap(
        z=cluster_similarity_matrix,
        x=cluster_labels,
        y=cluster_labels,
        colorscale='YlGnBu',
        zmin=0, zmax=1,
        colorbar=dict(tickvals=[0, 0.25, 0.5, 0.75, 1]),
    ))

    fig.update_layout(
        title='Within-Cluster Similarity',
        xaxis_title='Clusters',
        yaxis_title='Clusters',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return dcc.Graph(figure=fig)

# Function to generate between-cluster similarity heatmap
def update_between_cluster_heatmaps(island, similarity_type, protected_vars):
    cluster_similarity_matrix = compute_cluster_similarity(island, similarity_type, protected_vars)
    cluster_labels = [f'Cluster {i+1}' for i in range(len(island.get('clusters', {})))]

    fig = go.Figure(data=go.Heatmap(
        z=cluster_similarity_matrix,
        x=cluster_labels,
        y=cluster_labels,
        colorscale='YlGnBu',
        zmin=0, zmax=1,
        colorbar=dict(tickvals=[0, 0.25, 0.5, 0.75, 1]),
    ))

    fig.update_layout(
        title='Between-Cluster Similarity',
        xaxis_title='Clusters',
        yaxis_title='Clusters',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return dcc.Graph(figure=fig)

# Function for between-island heatmaps
def update_between_island_heatmaps(selected_island_data, similarity_type, protected_vars, island_data_list, selected_island):
    heatmap_figures = []
    for entry in island_data_list:
        if entry['island_index'] == selected_island:
            continue  # Skip the comparison of the island with itself

        island_b_data = entry['island_data']

        similarity_matrix, programs_a, programs_b = compute_similarity_between_islands(
            selected_island_data,
            island_b_data,
            similarity_type,
            protected_vars
        )

        # Add a fraction note to the title
        title = f'Island {selected_island} vs Island {entry["island_index"]} (showing {len(programs_a)} programs)'

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[f'Program A {i+1}' for i in range(len(programs_a))],
            y=[f'Program B {i+1}' for i in range(len(programs_b))],
            colorscale='YlGnBu',
            zmin=0, zmax=1,
            colorbar=dict(tickvals=[0, 0.25, 0.5, 0.75, 1]),
        ))
        fig.update_layout(
            title=title,
            title_font=dict(size=14),
            xaxis_title='Programs from Island A',
            yaxis_title='Programs from Island B',
            margin=dict(l=50, r=50, t=50, b=50),
        )

        heatmap_figures.append(html.Div(dcc.Graph(figure=fig), style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}))

    return html.Div(heatmap_figures, style={'text-align': 'center'})

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
