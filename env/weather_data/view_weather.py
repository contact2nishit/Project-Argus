import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx

# -----------------------
# CSV File
# -----------------------
DATA_CSV = os.path.join("env", "weather_data", "weather_snapshot_20251103_143543.csv")
df = pd.read_csv(DATA_CSV)

# grid_x = int(df["x"].max()) + 1
# grid_y = int(df["y"].max()) + 1
# grid_shape = (grid_x, grid_y)

# Create lat/lon grid for plotting
x_unique = np.sort(df["x"].unique())
y_unique = np.sort(df["y"].unique())
grid_shape = (len(y_unique), len(x_unique))

# -----------------------
# Dash App
# -----------------------
app = Dash(__name__)
app.title = "Weather Dashboard"

app.layout = html.Div([
    html.H2("Weather Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Button("Base Map", id="base", n_clicks=0, style={"margin": "5px"}),
        html.Button("Wind", id="wind", n_clicks=0, style={"margin": "5px"}),
        html.Button("Precipitation", id="precipitation", n_clicks=0, style={"margin": "5px"}),
        html.Button("Temperature", id="temperature", n_clicks=0, style={"margin": "5px"}),
        html.Button("Humidity", id="humidity", n_clicks=0, style={"margin": "5px"}),
        html.Button("Phase", id="phase", n_clicks=0, style={"margin": "5px"}),
        html.Button("Air Pressure", id="air_pressure", n_clicks=0, style={"margin": "5px"}),
        html.Label([
            dcc.Checklist(id='local', options=[{'label': 'Localize', 'value': 'checked'}], value=[])
        ])
    ], style={"textAlign": "center"}),

    dcc.Graph(id="map", style={"height": "80vh"}),

    # Store last selected map
    dcc.Store(id='current_map', data='base')
])

# -----------------------
# Helper: generate wind arrows
# -----------------------
def generate_wind_traces(df, arrow_scale=.1, step=1):
    traces = []
    annotations = []

    # Downsample to reduce clutter
    df_sub = df.iloc[::step, :]

    min_speed = df_sub["wind_speed"].min()
    max_speed = df_sub["wind_speed"].max()

    # Scale arrow length relative to map size
    x_span = df["x"].max() - df["x"].min()
    y_span = df["y"].max() - df["y"].min()
    arrow_length = min(x_span, y_span) * arrow_scale

    for _, row in df_sub.iterrows():
        x0, y0 = row["x"], row["y"]
        speed = row["wind_speed"]
        direction = row["wind_direction"]
        rad = np.radians(direction)

        x1 = x0 + arrow_length * np.sin(rad)
        y1 = y0 + arrow_length * np.cos(rad)

        color_frac = (speed - min_speed) / (max_speed - min_speed + 1e-6)
        color_rgb = f"rgb({int(50 + 100*color_frac)}, {int(150 + 50*color_frac)}, 255)"

        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor=color_rgb
        ))

    traces.append(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(colorscale=[[0,'lightblue'],[1,'darkblue']],
                    cmin=min_speed, cmax=max_speed, color=[min_speed,max_speed],
                    colorbar=dict(title="Wind Speed (m/s)"), size=0),
        showlegend=False, hoverinfo='none'
    ))

    return traces, annotations

# -----------------------
# Callback
# -----------------------
@app.callback(
    Output("map", "figure"),
    Output("current_map", "data"),
    Input("base", "n_clicks"),
    Input("wind", "n_clicks"),
    Input("precipitation", "n_clicks"),
    Input("temperature", "n_clicks"),
    Input("humidity", "n_clicks"),
    Input("phase", "n_clicks"),
    Input("air_pressure", "n_clicks"),
    Input("local", "value"),
    Input("current_map", "data")
)
def update_map(base, wind, precip, temp, hum, phase, air_pressure, local_value, current_map):
    triggered = ctx.triggered_id
    local_checked = 'checked' in (local_value or [])

    # Update map type if button clicked
    if triggered in ["base","wind","precipitation","temperature","humidity","phase","air_pressure"]:
        current_map = triggered

    # -----------------------
    # Generate figure
    # -----------------------
    if current_map == "base":
        fig = go.Figure(go.Heatmap(z=np.zeros((len(df["y"].unique()), len(df["x"].unique()))), colorscale="Greys"))
        fig.update_layout(title="Base Map", xaxis_title="Longitude", yaxis_title="Latitude")
        return fig, current_map

    elif current_map == "wind":
        traces, annotations = generate_wind_traces(df)
        fig = go.Figure(data=traces)

        # Limit figure to bbox
        min_x, max_x = df["x"].min(), df["x"].max()
        min_y, max_y = df["y"].min(), df["y"].max()
        pad_x = (max_x - min_x) * 0.02
        pad_y = (max_y - min_y) * 0.02

        fig.update_layout(
            title="Wind Vector Field",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            xaxis=dict(range=[min_x - pad_x, max_x + pad_x]),
            yaxis=dict(range=[min_y - pad_y, max_y + pad_y], scaleanchor="x"),
            annotations=annotations
        )
        return fig, current_map

    elif current_map in ["precipitation","temperature","humidity","air_pressure"]:
        col_map = {"precipitation":"precipitation","temperature":"temperature",
                   "humidity":"humidity","air_pressure":"air_pressure"}
        col = col_map[current_map]
        pivot = df.pivot_table(index="y", columns="x", values=col)
        x_coords = pivot.columns.values
        y_coords = pivot.index.values
        z_data = pivot.values

        # Default clamping and colors
        if current_map == "precipitation":
            zmin,zmax = 0,50
            colorscale=[[0.0,"lightgray"],[0.3,"lightblue"],[0.6,"blue"],[1.0,"purple"]]
            colorbar_title="Precipitation (mm)"
        elif current_map == "temperature":
            zmin,zmax = -30,50
            colorscale=[[0.0,"#2c7bb6"],[0.25,"#abd9e9"],[0.5,"#ffffbf"],[0.75,"#fdae61"],[1.0,"#d7191c"]]
            colorbar_title="Temperature (°C)"
        elif current_map == "humidity":
            zmin,zmax = 0,100
            colorscale=[[0.0,"#f7fcf5"],[0.25,"#c7e9c0"],[0.5,"#74c476"],[0.75,"#238b45"],[1.0,"#00441b"]]
            colorbar_title="Humidity (%)"
        elif current_map == "air_pressure":
            zmin,zmax = 980,1050
            colorscale=[[0.0,"#f7fbff"],[0.25,"#deebf7"],[0.5,"#9ecae1"],[0.75,"#3182bd"],[1.0,"#08519c"]]
            colorbar_title="Air Pressure (hPa)"

        if local_checked:
            zmin, zmax = np.nanmin(z_data), np.nanmax(z_data)

        fig = go.Figure(go.Heatmap(
            x=x_coords, y=y_coords, z=z_data,
            colorscale=colorscale, zmin=zmin, zmax=zmax,
            colorbar=dict(title=colorbar_title)
        ))

        fig.update_layout(title=f"{col.capitalize()} Map", xaxis_title="Longitude (°)", yaxis_title="Latitude (°)")
        fig.update_yaxes(autorange="reversed")

        return fig, current_map

    elif current_map == "phase":
        pivot = df.pivot_table(index="y", columns="x", values="phase", aggfunc=lambda s: s.mode()[0] if not s.empty else 0)
        z_data = pivot.values
        fig = go.Figure(go.Heatmap(
            x=pivot.columns.values, y=pivot.index.values,
            z=z_data, colorscale=["lightblue","white"], zmin=0, zmax=1,
            colorbar=dict(title="Phase")
        ))
        fig.update_layout(title="Precipitation Phase", xaxis_title="Longitude (°)", yaxis_title="Latitude (°)")
        fig.update_yaxes(autorange="reversed")
        return fig, current_map

    else:
        return go.Figure(), current_map

# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
