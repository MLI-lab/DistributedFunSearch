import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta
import time
import numpy as np
import plotly.express as px  # For generating a color scale
from similarity import compare_one_code_similarity_with_protection




# Initialize the Dash app
app = dash.Dash(__name__)

import os
import re

from datetime import datetime

def generate_slider_marks():
    timestamps = get_checkpoint_timestamps()
    return {
        i: datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%H:%M") 
        for i, ts in enumerate(timestamps)
    }



def get_checkpoint_timestamps():
    checkpoint_dir = os.path.join(os.getcwd(), "Checkpoints_old")
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    timestamps = [f.replace("checkpoint_", "").replace(".pkl", "") for f in files]
    return sorted(timestamps)

    # Extracting the timestamp from filenames like 'checkpoint_YYYY-MM-DD_HH-MM-SS.pkl'
    checkpoint_files = os.listdir(checkpoint_dir)
    timestamps = []
    for file in checkpoint_files:
        match = re.match(r"checkpoint_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.pkl", file)
        if match:
            timestamps.append(match.group(1))
    
    timestamps.sort()  # Sort by time
    return timestamps


def generate_performance_plot(timestamps, best_scores, title="Performance Over Time"):
    fig = go.Figure()

    # Add traces for each island, where y is the score for the island and x is the timestamp
    for i in range(len(best_scores)):
        fig.add_trace(go.Scatter(
            x=timestamps,  # Use the already formatted timestamps
            y=best_scores[i],  # Use the specific scores for each island
            mode='lines+markers',
            text=[str(score) for score in best_scores[i]],  # Display the scores on hover
            hoverinfo='text',
            name=f'Island {i+1}'
        ))

    # Adjust title layout for left alignment with more spacing
    fig.update_layout(
        xaxis_title="Time", 
        yaxis_title="Score", 
        title={
            'text': title,
            'x': 0.08,  # Left-align the title
            'xanchor': 'left'  # Anchor the title to the left
        },
        font=dict(family="Arial"),
        template="plotly_white",
        margin=dict(t=60, l=100)  # Add more left margin to push the title further right
    )
    return fig


def generate_cluster_plot(title, timestamps, cluster_data, color_cycle_threshold=20):
    """
    Generate a plot for clusters over time for a single island.

    Args:
    - title: Plot title (e.g., 'Island 1').
    - timestamps: List of full timestamps.
    - cluster_data: Dictionary where keys are full timestamps, and values are clusters at that timestamp.
    - color_cycle_threshold: Number of timestamps after which colors should cycle back (default: 20).

    Returns:
    - A Plotly figure.
    """
    fig = go.Figure()

    x_axis_timestamps = []
    y_scores = []
    marker_sizes = []
    hover_texts = []
    marker_colors = []

    # Define a custom color scale that skips yellow or light colors
    custom_colors = [
        '#440154', '#482878', '#3E4A89', '#31688E', '#26828E', '#1F9E89', 
        '#35B779', '#6DCD59', '#B4DE2C'  # Skipping lighter yellows
    ]

    num_colors_in_scale = len(custom_colors)

    # Track timestamp index to assign colors correctly
    for idx, ts in enumerate(timestamps):
        if ts in cluster_data:
            clusters = cluster_data[ts]
            if not clusters:
                continue  # Skip if there are no clusters at this timestamp

            cluster_keys = list(clusters.keys())
            sizes = [np.log10(len(clusters[key]['programs']) + 1) * 10 for key in cluster_keys]
            scores = [clusters[key]['score'] for key in cluster_keys]

            # Format timestamp for display
            formatted_time = datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%H:%M")

            # Spread clusters around the timestamp
            time_offsets = np.linspace(-1.5, 1.5, len(cluster_keys))
            timestamp_datetime = datetime.strptime(formatted_time, "%H:%M")
            spread_timestamps = [(timestamp_datetime + timedelta(minutes=offset)).strftime("%H:%M") for offset in time_offsets]

            x_axis_timestamps.extend(spread_timestamps)
            y_scores.extend(scores)
            marker_sizes.extend(sizes)
            hover_texts.extend([
                f"Signature: {key}<br>Programs: {len(clusters[key]['programs'])}<br>Score: {clusters[key]['score']}<br>Time: {formatted_time}"
                for key in cluster_keys
            ])

            # Cycle colors by using modulo to loop back to the start of the custom color scale
            for _ in cluster_keys:
                color_idx = idx % num_colors_in_scale
                marker_colors.append(custom_colors[color_idx])  # Assign the corresponding custom color to this cluster

        else:
            print(f"No data for timestamp {ts}")
            continue

    if not x_axis_timestamps:
        print(f"No data to plot for {title}")
        return go.Figure()

    # Prepare x-axis ticks
    formatted_timestamps = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%H:%M") for ts in timestamps]

    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=x_axis_timestamps,
        y=y_scores,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=marker_colors,  # Set the marker colors to our custom (cycled) color scale
            showscale=False  # Hide the color scale on the plot
        ),
        text=hover_texts,
        hoverinfo='text',
        showlegend=False
    ))

    # Update layout for the plot, including hover label customization
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cluster Score",
        title=title,
        font=dict(family="Arial"),
        template="plotly_white",
        height=190,
        width=300,
        margin=dict(l=0, r=0, t=45, b=2),
        xaxis=dict(
            tickmode='array',
            tickvals=formatted_timestamps,
            ticktext=formatted_timestamps
        ),
        hoverlabel=dict(
            font_size=10,  # Adjust the font size of the hover text here
            font_family="Arial"
        )
    )

    return fig





def load_cluster_data_from_checkpoint(checkpoint):
    """
    Load the cluster data from the given checkpoint. 
    Each island's state is a list of dictionaries with 'clusters'.
    
    Args:
    - checkpoint: Dictionary loaded from checkpoint file.
    
    Returns:
    - A list of cluster data for each island, where each island's cluster data is a list of dictionaries.
    """
    islands_state = checkpoint.get('islands_state', [])
    cluster_data_per_island = []

    # Iterate through each island's state
    for island_state in islands_state:
        clusters = island_state.get('clusters', {})
        cluster_list = []
        
        # Extract each cluster's programs and score
        for cluster_key, cluster_info in clusters.items():
            cluster_list.append({
                'programs': cluster_info.get('programs', []),
                'score': cluster_info.get('score', 0)
            })
        cluster_data_per_island.append(cluster_list)
    
    return cluster_data_per_island


# Layout of the app
app.layout = html.Div([
    html.H1("Evolutionary Progression", style={'text-align': 'center', 'font-family': 'Arial'}),

    # Two large boxes side by side
    html.Div([
        # First large box: Island selection
        html.Div([
            html.Div("Choose Islands:", style={'font-size': '14px', 'margin-top': '10px', 'margin-bottom': '27px', 'font-weight': 'bold', 'font-family': 'Arial'}),
            dcc.Checklist(
                id='island-options',
                options=[
                    {'label': 'Island 1', 'value': 'Island 1'},
                    {'label': 'Island 2', 'value': 'Island 2'},
                    {'label': 'Island 3', 'value': 'Island 3'}, 
                    {'label': 'Island 4', 'value': 'Island 4'}, 
                    {'label': 'Island 5', 'value': 'Island 5'},
                    {'label': 'Island 6', 'value': 'Island 6'},
                    {'label': 'Island 7', 'value': 'Island 7'},
                    {'label': 'Island 8', 'value': 'Island 8'},
                    {'label': 'Island 9', 'value': 'Island 9'},
                    {'label': 'Island 10', 'value': 'Island 10'}
                ],
                value=['Island 1', 'Island 2', 'Island 3', 'Island 4', 'Island 5', 
                        'Island 6', 'Island 7', 'Island 8', 'Island 9', 'Island 10'],  # Default value is all selected
                style={'font-size': '12px', 'display': 'grid', 'grid-template-columns': '1fr 1fr', 'gap': '3px', 'font-family': 'Arial'}
            )
        ], style={'width': '20%', 'padding': '10px', 'border': '1px solid black', 'box-shadow': '2px 2px 10px lightgray', 'margin-right': '10px'}),  # Adjusted width to 45%

        # Second large box: Slider and four smaller boxes
        html.Div([
            html.Div([
                dcc.RangeSlider(
                    min=0,
                    max=len(get_checkpoint_timestamps()) - 1,
                    step=1,
                    value=[0, 10],
                    marks=generate_slider_marks(),
                    id='time-range-slider-1',
                    included=True,
                    tooltip={'placement': 'top', 'always_visible': True}
                ),
                dcc.Interval(
                    id='interval-component',
                    interval=30*60*1000,  # 30 minutes in milliseconds
                    n_intervals=0
                ),
            ], style={'width': '100%', 'margin': '0 auto', 'margin-top': '20px'}),  # Full width for the slider

            # Four smaller boxes
            html.Div([
                html.Div([
                    html.Div("Reg. Programs:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", id='reg-programs', style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'}),
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'}),
            
                html.Div([
                    html.Div("Execution Errors:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", id='execution-errors', style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'}),
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'}),
            
                html.Div([
                    html.Div("Duplicates or VM:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", id='duplicates-vm', style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'}),
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'}),
            
                html.Div([
                    html.Div("Last Reset Time:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", id='last-reset-time', style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'}),
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'margin-top': '20px'})
        ], style={'width': '80%', 'padding': '10px', 'border': '1px solid black', 'box-shadow': '2px 2px 10px lightgray'})  # Adjusted width to 55%
    ], style={'display': 'flex', 'width': '83.3%', 'justify-content': 'space-between', 'margin-top': '30px', 'margin-bottom': '20px', 'margin-right': '0px','margin-left': '97px' }),


    # Performance Graph
    html.Div([
        dcc.Graph(
            id='performance-graph',
            figure=generate_performance_plot([], [], "Performance Over Time")  # Adding the title
        )
    ], style={'border': '1px solid black', 'padding': '10px', 'width': '81.5%', 'margin-right': '0px','margin-left': '97px'}),

    # New section for Cluster Scores + Size Over Time
    html.Div([
        html.H2("Cluster Scores + Size Over Time", 
            style={'text-align': 'center', 'font-weight': 'bold', 'padding': '0px', 'font-size': '28px', 'font-family': 'Arial'}), 

        html.Div([
            # Dropdown for selecting start time
            html.Div([
                html.Label("Select Start Time", style={'font-weight': 'bold', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='start-time-dropdown',
                    options=[{'label': datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%m-%d %H:%M"), 'value': i}
                            for i, ts in enumerate(get_checkpoint_timestamps())],  # Use formatted timestamps as options
                    value=0,  # Default to the earliest time
                    clearable=False,
                    style={'font-size': '14px', 'margin-top': '10px', 'font-family': 'Arial'}
                )
            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%','margin-top': '0px', 'margin-bottom': '25px'}),

            # Dropdown for selecting end time
            html.Div([
                html.Label("Select End Time", style={'font-weight': 'bold', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='end-time-dropdown',
                    options=[{'label': datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%m-%d %H:%M"), 'value': i}
                            for i, ts in enumerate(get_checkpoint_timestamps())],  # Use formatted timestamps as options
                    value=len(get_checkpoint_timestamps()) - 1,  # Default to the latest time
                    clearable=False,
                    style={'font-size': '14px', 'margin-top': '10px', 'font-family': 'Arial'}
                )
            ], style={'width': '45%', 'display': 'inline-block'})
        ], style={'margin-top': '10px', 'text-align': 'center'}),


        # First row of plots (3 plots)
        html.Div([
            html.Div([dcc.Graph(id='island-1-plot')], style={'width': '31.4%', 'height': '200px'}),
            html.Div([dcc.Graph(id='island-2-plot')], style={'width': '31.4%', 'height': '200px'}),
            html.Div([dcc.Graph(id='island-3-plot')], style={'width': '31.4%', 'height': '200px'})
        ], style={'display': 'flex', 'justify-content': 'center'}),

        # Second row of plots (3 plots)
        html.Div([
            html.Div([dcc.Graph(id='island-4-plot')], style={'width': '31.4%', 'height': '200px'}),
            html.Div([dcc.Graph(id='island-5-plot')], style={'width': '31.4%', 'height': '200px'}),
            html.Div([dcc.Graph(id='island-6-plot')], style={'width': '31.4%', 'height': '200px'})
        ], style={'display': 'flex', 'justify-content': 'center'}),

        # Third row of plots (3 plots)
        html.Div([
            html.Div([dcc.Graph(id='island-7-plot')], style={'width': '31.4%', 'height': '200px'}),
            html.Div([dcc.Graph(id='island-8-plot')], style={'width': '31.4%', 'height': '200px'}),
            html.Div([dcc.Graph(id='island-9-plot')], style={'width': '31.4%', 'height': '200px'})
        ], style={'display': 'flex', 'justify-content': 'center'}),

        # Fourth row of plots (1 plot)
        html.Div([
            html.Div([dcc.Graph(id='island-10-plot')], style={'width': '31.4%', 'height': '200px', 'margin-left': '29px'}),
        ], style={'display': 'flex'}),
    ], style={'border': '1px solid black', 'padding': '20px', 'width': '80%', 'margin-top': '20px', 'margin-bottom': '20px','margin-right': '0px','margin-left': '97px'}),



    # Program Similarity section
    html.Div(id='program-similarity-section', children=[
        html.H2("Program Similarity", style={'text-align': 'center', 'font-weight': 'bold', 'padding': '10px', 'font-size': '28px', 'font-family': 'Arial'}),

        # This is where the description (e.g., "Computing similarities...") will be shown
        html.Div(id='similarity-description', style={'text-align': 'center', 'font-size': '20px', 'font-family': 'Arial', 'margin-top': '20px'}),

        # Two large boxes side by side
        html.Div([
            # First large box: Island selection using radio buttons
            html.Div([
                html.Div("Choose Island:", style={'font-size': '14px', 'margin-top': '10px', 'font-weight': 'bold', 'font-family': 'Arial'}),
                dcc.RadioItems(
                    id='island-options-2',
                    options=[{'label': f'Island {i+1}', 'value': i} for i in range(10)],  # Island 1 to 10
                    value=0,  # Default to Island 1
                    labelStyle={'display': 'block'},  # Display options vertically
                    style={'font-size': '14px', 'font-family': 'Arial', 'display': 'grid', 'grid-template-columns': '1fr 1fr', 'grid-gap': '10px', 'margin-top': '10px'}
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid black', 'box-shadow': '2px 2px 10px lightgray', 'margin-right': '10px', 'margin-top': '10px'}),  # Left large box

            # Second large box: Granularity options and timestamp selection
            html.Div([
                # Similarity Granularity options
                html.Div([
                    html.Div("Choose Similarity Granularity:", style={'font-size': '14px', 'margin-top': '10px', 'font-weight': 'bold', 'font-family': 'Arial'}),
                    dcc.RadioItems(
                        id='granularity-options',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Within Cluster', 'value': 'within_cluster'},
                            {'label': 'Between Cluster', 'value': 'between_cluster'},
                            {'label': 'Between Islands', 'value': 'between_islands'}
                        ],
                        value='none',  # Default value
                        labelStyle={'display': 'block'},  # Display options vertically
                        style={'font-size': '14px', 'font-family': 'Arial', 'margin-top': '10px'}
                    )
                ], style={'margin-top': '10px', 'text-align': 'left'}),

                # Dropdown for selecting timestamp
                html.Div([
                    html.Label("Select Timestamp", style={'font-size': '14px', 'margin-top': '10px', 'font-weight': 'bold', 'font-family': 'Arial'}),
                    dcc.Dropdown(
                        id='timestamp-dropdown',
                        options=[{'label': datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%m-%d %H:%M"), 'value': i}
                                for i, ts in enumerate(get_checkpoint_timestamps())],  # Use formatted timestamps as options
                        value=0,  # Default to the earliest time
                        clearable=False,
                        style={'font-size': '14px', 'margin-top': '30px', 'font-family': 'Arial'}
                    )
                ], style={'width': '100%', 'margin-top': '10px', 'margin-bottom': '25px'}),
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid black', 'box-shadow': '2px 2px 10px lightgray', 'margin-right': '10px', 'margin-top': '10px'}),  # Right large box
        ], style={'display': 'flex', 'width': '100%', 'justify-content': 'space-between', 'margin-top': '30px', 'margin-bottom': '20px', 'margin-right': '0px', 'margin-left': '0px'}),

        # Use dcc.Loading to show a spinner while both outputs are being computed
        dcc.Loading(
            id="loading-heatmap",
            type="default",  # You can change this to 'circle' or other types
            children=html.Div([
                html.Div(id='similarity-description', style={
                    'text-align': 'center', 'font-size': '20px', 'font-family': 'Arial', 'margin-top': '20px'
                }),
                html.Div(id='heatmap-container', style={'margin-top': '20px'})
            ])
        )
    ], style={'padding': '20px', 'width': '85%', 'margin': 'auto', 'margin-top': '20px', 'margin-right': '0px', 'margin-left': '80px'})


])# end closing with first Div 

########################################Backend Logic#####################################

def load_checkpoint(timestamp_index):
    timestamps = get_checkpoint_timestamps()
    if timestamp_index < len(timestamps):
        timestamp = timestamps[timestamp_index]
        filepath = os.path.join("Checkpoints", f"checkpoint_{timestamp}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            print(f"Checkpoint file {filepath} not found!")
    return None

################################## Callback Logic for cluster scores and sizes over time ###################################
@app.callback(
    [Output(f'island-{i+1}-plot', 'figure') for i in range(10)],  # Outputs for each island plot
    [Input('start-time-dropdown', 'value'),
     Input('end-time-dropdown', 'value')]
)

def update_cluster_subplots(start_time_idx, end_time_idx):
    # Get all available timestamps
    timestamps = get_checkpoint_timestamps()

    # Ensure the selected range is within bounds
    start_idx = min(len(timestamps) - 1, start_time_idx)
    end_idx = min(len(timestamps) - 1, end_time_idx)

    # Get the relevant timestamps (full timestamps)
    selected_timestamps = timestamps[start_idx:end_idx+1]

    # Initialize data structures for each island
    cluster_data_per_island = [{} for _ in range(10)]  # List of dictionaries for each island

    for idx in range(len(selected_timestamps)):
        checkpoint = load_checkpoint(start_idx + idx)  # Using index to load checkpoint
        if checkpoint is not None:
            # Assuming clusters are stored under 'islands_state' in checkpoint data
            islands_state = checkpoint.get('islands_state', [])
            for i in range(10):  # For each island
                if i < len(islands_state):
                    island_data = islands_state[i]  # Should be a dict with 'clusters' key
                    if 'clusters' in island_data:
                        clusters = island_data['clusters']
                        # Add clusters under the timestamp key
                        if clusters:
                            cluster_data_per_island[i][selected_timestamps[idx]] = clusters
                        else:
                            print(f"No clusters for island {i+1} at timestamp {selected_timestamps[idx]}")
                    else:
                        print(f"No 'clusters' key in island_data for island {i+1} at timestamp {selected_timestamps[idx]}")
                else:
                    print(f"No data for island {i+1} at timestamp {selected_timestamps[idx]}")
        else:
            print(f"Checkpoint missing for timestamp {selected_timestamps[idx]}")

    # Generate a plot for each island
    plots = []
    for i in range(10):
        cluster_data = cluster_data_per_island[i]  # Dictionary of full timestamps to clusters
        if not cluster_data:
            plots.append(go.Figure())  # No data for this island
            continue
        try:
            # Pass full timestamps to generate_cluster_plot
            plot = generate_cluster_plot(f"Island {i+1}", selected_timestamps, cluster_data)
            plots.append(plot)
        except Exception as e:
            print(f"Error generating plot for Island {i+1}: {e}")
            plots.append(go.Figure())

    return plots

######################################### Callback logic for adjusting the four little boxes #################################################
@app.callback(
    [Output('time-range-slider-1', 'marks'),
     Output('time-range-slider-1', 'max')],
    [Input('interval-component', 'n_intervals')]
)
def update_slider_marks(n_intervals):
    # Retrieve the latest checkpoint timestamps and update the slider
    timestamps = get_checkpoint_timestamps()
    marks = generate_slider_marks()  # Recalculate marks based on the checkpoint files
    
    return marks, len(timestamps) - 1

@app.callback(
    [Output('reg-programs', 'children'),
     Output('execution-errors', 'children'),
     Output('duplicates-vm', 'children'),
     Output('last-reset-time', 'children')],
    [Input('time-range-slider-1', 'value')]
)
def update_checkpoint_values(time_range):
    # Check if the left and right values are equal (single timestamp)
    if isinstance(time_range, list) and time_range[0] == time_range[1]:
        # Single timestamp selected
        checkpoint = load_checkpoint(time_range[0])
        
        if checkpoint is None:
            return "N/A", "N/A", "N/A", "N/A"
        
        # Values for a single checkpoint
        reg_programs = checkpoint.get('registered_programs', "N/A")
        exec_errors = checkpoint.get('execution_failed', "N/A")
        duplicates_vm = checkpoint.get('total_programs', 0) - checkpoint.get('registered_programs', 0)
        last_reset_time = time.strftime("%H:%M", time.localtime(checkpoint.get('last_reset_time', 0)))

    else:
        # Range of timestamps selected
        start_time = time_range[0]
        end_time = time_range[1]
        start_checkpoint = load_checkpoint(start_time)
        end_checkpoint = load_checkpoint(end_time)

        if start_checkpoint is None or end_checkpoint is None:
            return "N/A", "N/A", "N/A", "N/A"

        # Calculate differences for range selection
        reg_programs = end_checkpoint['registered_programs'] - start_checkpoint['registered_programs']
        exec_errors = end_checkpoint['execution_failed'] - start_checkpoint['execution_failed']
        duplicates_vm = (end_checkpoint['total_programs'] - end_checkpoint['registered_programs']) - \
                        (start_checkpoint['total_programs'] - start_checkpoint['registered_programs'])
        last_reset_time = time.strftime("%H:%M", time.localtime(end_checkpoint['last_reset_time']))

    return reg_programs, exec_errors, duplicates_vm, last_reset_time

################################################## Callback Logic for Performance over time Plot ########################
@app.callback(
    Output('performance-graph', 'figure'),
    [Input('island-options', 'value'),
     Input('time-range-slider-1', 'value')]
)
def update_performance_plot(selected_islands, selected_time_range):
    timestamps = get_checkpoint_timestamps()

    # Ensure the selected range is within the bounds of available timestamps
    start_idx = min(len(timestamps) - 1, selected_time_range[0])
    end_idx = min(len(timestamps) - 1, selected_time_range[1])

    # Get the relevant timestamps
    selected_timestamps = timestamps[start_idx:end_idx+1]

    # Format timestamps to display only hour and minute
    formatted_timestamps = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").strftime("%H:%M") for ts in selected_timestamps]

    # Load checkpoints and gather best_scores_per_island for the range
    best_scores = [[] for _ in range(10)]  # Assuming 10 islands
    
    for i in range(start_idx, end_idx + 1):
        checkpoint = load_checkpoint(i)
        if checkpoint is not None:
            for j in range(10):  # Iterate through each island's scores
                best_scores[j].append(checkpoint['best_score_per_island'][j])
        else:
            for j in range(10):
                best_scores[j].append(0)  # Default if checkpoint is missing

    # Generate the plot with formatted timestamps and specific scores for each island
    fig = generate_performance_plot(formatted_timestamps, best_scores)

    # Update visibility of islands based on the checklist selection
    for i, trace in enumerate(fig.data):
        island_name = f'Island {i+1}'
        trace.visible = island_name in selected_islands

    return fig

#########################################Similarity Plot Logic#################################################
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc

def update_within_cluster_heatmaps(island_data, similarity_type, protected_vars):
    clusters = island_data.get('clusters', {})
    
    if not clusters:
        return html.Div("No clusters found.")

    cluster_keys = list(clusters.keys())
    heatmaps = []

    # Iterate through each cluster in the island
    for cluster_key in cluster_keys:
        programs = clusters[cluster_key]['programs']
        n_programs = len(programs)
        
        if n_programs == 0:
            continue

        # Initialize a similarity matrix (programs x programs)
        similarity_matrix = np.zeros((n_programs, n_programs))

        # Compare programs within the cluster
        for i in range(n_programs):
            for j in range(i, n_programs):
                similarity = compare_one_code_similarity_with_protection(
                    programs[i], programs[j], similarity_type, protected_vars
                )

                # Check for NaN values in the similarity
                if np.isnan(similarity):
                    print(f"NaN value encountered for programs {i+1} and {j+1} in cluster {cluster_key}")
                    return html.Div(f"Got invalid similarity value (NaN) for programs {i+1} and {j+1} in cluster {cluster_key}.")

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric matrix

        # Create the heatmap figure
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[f'Program {i+1}' for i in range(n_programs)],
            y=[f'Program {i+1}' for i in range(n_programs)],
            colorscale='YlGnBu',
            zmin=0, zmax=1,
            colorbar=dict(
                thickness=10,  # Thinner color bar
                len=1.1,  # Reduce length if needed
                tickfont=dict(size=8)  # Smaller tick labels for color bar
            )
        ))

        # Update layout to ensure the labels are rotated and the title is smaller
        fig.update_layout(
            width=300,  
            height=300,  
            title=dict(
                text=f'Cluster: {cluster_key} ',
                font=dict(size=10),  # Smaller title font
                x=0.5,  # Center the title
                xanchor='center'
            ),
            xaxis=dict(
                tickangle=-90,  # Rotate program labels vertically
                tickfont=dict(size=5)  # Smaller font for x-axis labels
            ),
            yaxis=dict(
                tickfont=dict(size=5),  # Smaller font for y-axis labels
                automargin=True  # Ensure proper margin for rotated labels
            ),
            margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins as needed
        )

        heatmaps.append(dcc.Graph(figure=fig))

    # Return a Div containing all heatmaps
    return html.Div(heatmaps, style={'display': 'flex', 'flex-wrap': 'wrap'})



def update_between_cluster_heatmaps(island_data, similarity_type, protected_vars):
    clusters = island_data.get('clusters', {})
    
    if not clusters:
        return html.Div("No clusters found.")

    cluster_keys = list(clusters.keys())
    n_clusters = len(cluster_keys)

    # Initialize a similarity matrix (clusters x clusters)
    similarity_matrix = np.zeros((n_clusters, n_clusters))

    hover_text = []  # This will store the custom hover text for each cell

    # Compare clusters with each other
    for i, cluster_a_key in enumerate(cluster_keys):
        hover_row = []  # This will store the hover text for each row
        for j, cluster_b_key in enumerate(cluster_keys):
            programs_a = clusters[cluster_a_key]['programs']
            programs_b = clusters[cluster_b_key]['programs']
            
            max_similarity = 0
            if programs_a and programs_b:
                # Compare all programs in cluster A with all programs in cluster B
                for prog_a in programs_a:
                    for prog_b in programs_b:
                        similarity = compare_one_code_similarity_with_protection(
                            prog_a, prog_b, similarity_type, protected_vars
                        )
                        max_similarity = max(max_similarity, similarity)
            
            similarity_matrix[i, j] = max_similarity

            # Create the custom hover text
            hover_text_entry = (
                f"Cluster {i+1} {cluster_a_key}<br>"
                f"Cluster {j+1} {cluster_b_key}<br>"
                f"Similarity: {max_similarity:.2f}"
            )
            hover_row.append(hover_text_entry)

        hover_text.append(hover_row)

    # Create the heatmap figure with custom hover text
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f'Cluster {i+1}' for i in range(n_clusters)],
        y=[f'Cluster {i+1}' for i in range(n_clusters)],
        colorscale='Blues',
        zmin=0, zmax=1,
        colorbar=dict(len=1.05),
        hoverinfo="text",  # Use custom hover text
        hovertext=hover_text  # Set custom hover text
    ))

    # Update layout for the heatmap
    fig.update_layout(
        title='Between-Cluster Similarity',
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(tickangle=45),  # Rotate x-axis labels for better readability
        coloraxis_showscale=False  # Optional: Adjust size of the color bar
    )

    return dcc.Graph(figure=fig)



def update_between_island_heatmaps(island_data_a, similarity_type, protected_vars, all_islands_data, selected_island_index):
    import traceback  # Import traceback to print detailed exception info
    heatmap_figures = []

    # Loop through all other islands to compare with the selected island
    for idx, island_data_b in enumerate(all_islands_data):
        if idx == selected_island_index:
            continue  # Skip comparison with the same island

        programs_a = []
        programs_b = []
        cluster_a_signatures = []
        cluster_b_signatures = []

        # Collect all programs from island A and island B, as well as their cluster signatures
        for cluster_a_key, cluster_a in island_data_a.get('clusters', {}).items():
            programs_a.extend(cluster_a.get('programs', []))
            cluster_a_signatures.extend([cluster_a_key] * len(cluster_a.get('programs', [])))

        for cluster_b_key, cluster_b in island_data_b.get('clusters', {}).items():
            programs_b.extend(cluster_b.get('programs', []))
            cluster_b_signatures.extend([cluster_b_key] * len(cluster_b.get('programs', [])))

        # If there are no programs to compare, skip this island comparison
        if not programs_a or not programs_b:
            continue

        # Compute similarity between all programs in island A and B
        similarity_scores = []
        for i, prog_a in enumerate(programs_a):
            for j, prog_b in enumerate(programs_b):
                try:
                    similarity = compare_one_code_similarity_with_protection(
                        prog_a, prog_b, similarity_type, protected_vars
                    )
                    similarity_scores.append((similarity, cluster_a_signatures[i], cluster_b_signatures[j]))
                except Exception as e:
                    print(f"Error comparing programs at indices {i} (Island {selected_island_index + 1}) and {j} (Island {idx + 1}): {e}")
                    print(f"Program A (index {i}):\n{prog_a}\n")
                    print(f"Program B (index {j}):\n{prog_b}\n")
                    traceback.print_exc()  # Print the full traceback
                    continue  # Skip this pair

        # If no similarity scores were computed, skip to next island
        if not similarity_scores:
            continue

        # Sort the similarity scores to find the top 10 most similar program pairs
        similarity_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity score (highest first)
        top_10_similarities = similarity_scores[:10]  # Keep only the top 10 similarities

        # Adjust labels in case there are fewer than 10 similarities
        num_similarities = len(top_10_similarities)
        programs_a_labels = [f'Program {i+1}' for i in range(num_similarities)]
        programs_b_labels = ['Top Similar Programs']

        # Extract similarity values and reshape to match heatmap dimensions
        top_10_similarity_matrix = np.array([sim[0] for sim in top_10_similarities]).reshape((num_similarities, 1))

        # Reshape hover text to match the shape of z
        hover_text_matrix = [[
            f'Cluster Signature: {sim[1]}<br>Cluster Signature: {sim[2]}<br>Similarity: {sim[0]:.2f}'
        ] for sim in top_10_similarities]

        # Generate the heatmap for this island comparison
        fig = go.Figure(data=go.Heatmap(
            z=top_10_similarity_matrix,
            x=programs_b_labels,
            y=programs_a_labels,
            colorscale='Reds',
            zmin=0, zmax=1,
            colorbar=dict(thickness=10),
            hoverinfo='text',
            hovertext=hover_text_matrix  # Use the reshaped hover text
        ))

        fig.update_layout(
            title=f'Island {selected_island_index + 1} vs Island {idx + 1}',
            height=300,
            width=300,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(tickfont=dict(size=10)),  # Adjust font size if needed
            yaxis=dict(tickfont=dict(size=10))
        )

        heatmap_figures.append(html.Div(
            dcc.Graph(figure=fig),
            style={'width': '300px', 'height': '300px', 'display': 'inline-block', 'margin': '10px'}
        ))

    # Return the final heatmaps
    return html.Div(
        heatmap_figures,
        style={'display': 'grid', 'grid-template-columns': 'repeat(3, 1fr)', 'gap': '20px', 'justify-content': 'center'}
    )

##################################Similarity Measure Callback Logic ########################################
@app.callback(
    [Output('similarity-description', 'children'),
     Output('heatmap-container', 'children')],
    [Input('granularity-options', 'value'),
     Input('timestamp-dropdown', 'value'),
     Input('island-options-2', 'value')]
)
def update_similarity_plots(selected_granularity, selected_timestamp_idx, selected_island_idx):
    try:
        # Set description to indicate computation is in progress
        description = ""

        # Retrieve the available timestamps
        timestamps = get_checkpoint_timestamps()

        # Validate selected_timestamp_idx
        if isinstance(selected_timestamp_idx, int) and 0 <= selected_timestamp_idx < len(timestamps):
            selected_timestamp = timestamps[selected_timestamp_idx]
        else:
            return "Invalid timestamp selection", html.Div("No data available.")

        # Load the checkpoint data for the selected timestamp
        checkpoint_data = load_checkpoint(selected_timestamp_idx)

        if not checkpoint_data or 'islands_state' not in checkpoint_data:
            return "No data available.", html.Div("No data available.")

        # Get the list of islands and select the island based on the chosen index
        island_states = checkpoint_data.get('islands_state', [])
        if selected_island_idx < len(island_states):
            selected_island_data = island_states[selected_island_idx]
        else:
            return f"Invalid island selection. Only {len(island_states)} islands available.", html.Div("No data available.")

        # Generate heatmaps based on granularity
        if selected_granularity == 'within_cluster':
            heatmap = update_within_cluster_heatmaps(selected_island_data, 'bag_of_nodes', ['node', 'G', 'n', 's'])
        elif selected_granularity == 'between_cluster':
            heatmap = update_between_cluster_heatmaps(selected_island_data, 'bag_of_nodes', ['node', 'G', 'n', 's'])
        elif selected_granularity == 'between_islands':
            # Pass all island states to compare with the selected island
            heatmap = update_between_island_heatmaps(selected_island_data, 'bag_of_nodes', ['node', 'G', 'n', 's'], island_states, selected_island_idx)
        else:
            return "No granularity selected", html.Div("Please select a granularity option to view similarity.")

        # Return the final outputs: description and heatmap
        return description, heatmap

    except Exception as e:
        print(f"Exception: {e}")
        return "An error occurred.", html.Div("An error occurred.")





#########################Formating dynamic layout##########################
# New layout for plot-container using CSS Grid
def generate_plot_container(plots):
    wrapped_plots = [
        html.Div(
            children=plot,
            style={
                'border': '1px solid black',
                'padding': '2px',
                'background-color': 'white',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'width': '100%',  # Flexible width within the grid cell
                'height': 'auto',  # Ensure the height adjusts automatically
                'margin': '10px'  # Adds some space between plots
            }
        ) for plot in plots
    ]
    
    return html.Div(
        children=wrapped_plots,
        style={
            'display': 'grid',
            'grid-template-columns': 'repeat(auto-fit, minmax(300px, 1fr))',  # Adjusts based on available space
            'gap': '20px',
            'width': '100%',
            'align-items': 'center',
            'justify-content': 'center',
            'background-color': 'white',
            'margin': '0 auto',  # Center the whole container
        }
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)