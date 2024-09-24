import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Helper function to generate placeholder plots
def generate_placeholder_plot(title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode='lines+markers'))
    fig.update_layout(
        xaxis_title="Time", 
        yaxis_title="Score", 
        title=title,
        font=dict(family="Arial"),
        template="plotly_white"  # Applying the same layout template for consistency
    )
    return fig


def generate_placeholder_dynplot(title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode='lines+markers'))
    fig.update_layout(
        xaxis_title="Time", 
        yaxis_title="Score", 
        title=title,
        font=dict(family="Arial"),
        template="plotly_white",  # Applying the same layout template for consistency
        height=250,  # Adjust this to control plot height
        width=300,   # Adjust this to control plot width
        margin=dict(l=10, r=10, t=45, b=10)  # Adjust the margins around the plot
    )
    return fig

def generate_placeholder_subplot(title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode='lines+markers'))
    fig.update_layout(
        xaxis_title="Time", 
        yaxis_title="Score", 
        title=title,
        font=dict(family="Arial"),
        template="plotly_white",  # Applying the same layout template for consistency
        height=190,  # Adjust this to control plot height
        width=300,   # Adjust this to control plot width
        margin=dict(l=0, r=0, t=45, b=2)  # Adjust the margins around the plot
    )
    return fig

# Layout of the app
app.layout = html.Div([
    html.H1("Evolutionary Progression", style={'text-align': 'center', 'font-family': 'Arial'}),

    # Two large boxes side by side
    html.Div([
        # First large box: Slider and granularity options
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
                    style={'font-size': '14px', 'margin-top': '10px', 'font-family': 'Arial'}
                )
            ], style={'margin-top': '10px', 'text-align': 'left'})
        ], style={'flex': 1, 'padding': '10px', 'border': '1px solid black', 'box-shadow': '2px 2px 10px lightgray', 'margin-right': '10px'}),  # First large box

        # Second large box: Same slider and four smaller boxes
        html.Div([
            # Slider
            dcc.Slider(
                min=0,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(21)},
                id='time-range-slider-2',  # Second slider ID
                included=True,
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
            # Four smaller boxes
            html.Div([
                html.Div([
                    html.Div("Reg. Programs:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'})
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'}), # Adjusted width to fit 4 boxes
                
                html.Div([
                    html.Div("Execution Errors:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'})
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Div("Duplicates or VM:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'})
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Div("Last Reset Time:", style={'font-size': '12px', 'text-align': 'center', 'font-family': 'Arial'}),
                    html.Div("0", style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center', 'font-family': 'Arial'})
                ], style={
                    'border': '1px solid black', 'padding': '10px', 'margin': '5px', 'box-shadow': '2px 2px 10px lightgray',
                    'text-align': 'center', 'width': '23%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'margin-top': '50px'})
        ], style={'flex': 1, 'padding': '10px', 'border': '1px solid black', 'box-shadow': '2px 2px 10px lightgray'})
    ], style={'display': 'flex', 'width': '83.3%', 'margin': 'auto', 'justify-content': 'space-between', 'margin-top': '30px', 'margin-bottom': '20px'}),

    # Performance Graph
    html.Div([
        dcc.Graph(
            id='performance-graph',
            figure=generate_placeholder_plot("Performance Over Time")  # Adding the title
        )
    ], style={'border': '1px solid blue', 'padding': '20px', 'margin-top': '20px', 'width': '80%', 'margin': 'auto'}),

    # New section for Cluster Scores + Size Over Time
    html.Div([
        html.H2("Cluster Scores + Size Over Time", 
            style={'text-align': 'center', 'font-weight': 'bold', 'padding': '10px', 'font-size': '28px', 'font-family': 'Arial'}), 

        # First row of plots (3 plots)
        html.Div([
            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 1"))  # Using the consistent plot layout
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),

            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 2"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),

            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 3"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),
        ],style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '0px'}),

        # Second row of plots (3 plots)
        html.Div([
            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 4"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),

            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 5"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),

            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 6"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '0px'}),

        # Third row of plots (3 plots)
        html.Div([
            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 7"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),

            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 8"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),

            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 9"))
            ], style={'width': '40%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '0px'}),

        # Fourth row of plots (1 plot)
        html.Div([
            html.Div([
                dcc.Graph(figure=generate_placeholder_subplot("Island 10"))
            ], style={'width': '29.5%', 'height': '200px', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'box-shadow': '2px 2px 10px lightgray', 'margin': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '0px'}),

    ], style={'margin-top': '0px', 'width': '84.4%', 'margin': 'auto', 'margin-bottom': '1px'}),



    # Program Similarity section
    html.Div(id='program-similarity-section', children=[
        html.H2("Program Similarity", style={'text-align': 'center', 'font-weight': 'bold', 'padding': '10px', 'font-size': '28px', 'font-family': 'Arial'}),
        html.Div(id='similarity-description', style={'text-align': 'center', 'font-size': '20px', 'font-family': 'Arial', 'margin-top': '20px'}),

    # Slider for time stamps
    html.Div([
        dcc.Slider(
            id='time-slider',
            min=0,
            max=10,  # Example: 10 time stamps available
            step=1,
            value=5,  # Default value
            marks={i: f'Time {i}' for i in range(11)},  # Labels for time points
            tooltip={'always_visible': True},
            updatemode='drag'  # Update while dragging
        )
    ], style={'width': '100%', 'margin': 'auto', 'margin-top': '20px'}),
        
        # Placeholder for dynamic plot container
        html.Div(id='plot-container', style={'margin-top': '20px'})
    ], style={ 'padding': '20px', 'width': '82%', 'margin': 'auto', 'margin-top': '20px'})
])

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
                'justify-content': 'center'
            }
        ) for plot in plots
    ]
    
    return html.Div(
        children=wrapped_plots,
        style={
            'display': 'grid',
            'grid-template-columns': 'repeat(auto-fill, minmax(330px, 1fr))',
            'gap': '20px',
            'width': '100%',
            'align-items': 'center',
            'justify-content': 'center',
            'background-color': 'white'
        }
    )

# Callback to update both the description and the plot container
@app.callback(
    [Output('similarity-description', 'children'),
     Output('plot-container', 'children')],
    [Input('granularity-options', 'value'),
     Input('time-slider', 'value')]  # Now also listens to slider value
)
def update_similarity_plots(selected_granularity, selected_time):
    # Logic to select different plots based on granularity option and slider value
    description = f"Program Similarity at Time {selected_time}"
    
    if selected_granularity == 'within_cluster':
        plots = [dcc.Graph(figure=generate_placeholder_dynplot(f"Within Cluster Plot (Time {selected_time})"))]

    elif selected_granularity == 'between_cluster':
        plots = [
            dcc.Graph(figure=generate_placeholder_dynplot(f"Between Cluster 1 (Time {selected_time})")),
            dcc.Graph(figure=generate_placeholder_dynplot(f"Between Cluster 2 (Time {selected_time})"))
        ]

    elif selected_granularity == 'between_islands':
        plots = [
            dcc.Graph(figure=generate_placeholder_dynplot(f"Between Islands 1 (Time {selected_time})")),
            dcc.Graph(figure=generate_placeholder_dynplot(f"Between Islands 2 (Time {selected_time})")),
            dcc.Graph(figure=generate_placeholder_dynplot(f"Between Islands 3 (Time {selected_time})"))
        ]
    else:
        description = "No Program Similarity Selected"
        plots = []

    # Generate the grid container with dynamic number of plots
    plot_container = generate_plot_container(plots)

    return description, plot_container

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)