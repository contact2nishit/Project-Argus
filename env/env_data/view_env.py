import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from scipy.interpolate import griddata

# -----------------------
# CSV File
# -----------------------
DATA_CSV = os.path.join("env", "env_data", "env_snapshot.csv")
df = pd.read_csv(DATA_CSV)
df = df.sort_values(["latitude", "longitude"]).dropna(subset=["latitude", "longitude"])

selected_date = df["date"].iloc[0]
df = df[df["date"] == selected_date].copy()

print(f"Displaying data for {selected_date}")
# -----------------------
# Dash App
# -----------------------
app = Dash(__name__)
app.title = "Weather Dashboard"

map_options = {
    0: "base",
    1: "wind_10m",
    2: "wind_80m",
    3: "precipitation",
    4: "temperature_2m",
    5: "temperature_80m",
    6: "relative_humidity",
    7: "surface_pressure",
    8: "cloud_cover_low",
    9: "visibility"
}

marks = {
    0: {"label": "Terrain", "style": {"color": "gray"}},
    1: {"label": "Wind 10m", "style": {"color": "gray"}},
    2: {"label": "Wind 80m", "style": {"color": "gray"}},
    3: {"label": "Precipitation", "style": {"color": "gray"}},
    4: {"label": "Temperature 2m", "style": {"color": "gray"}},
    5: {"label": "Temperature 80m", "style": {"color": "gray"}},
    6: {"label": "Relative Humidity", "style": {"color": "gray"}},
    7: {"label": "Surface Pressure", "style": {"color": "gray"}},
    8: {"label": "Cloud Cover Low", "style": {"color": "gray"}},
    9: {"label": "Visibility", "style": {"color": "gray"}},
}

app.layout = html.Div([
    html.H2("Weather Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            dcc.Graph(id="map", style={"height": "85vh", "width": "85vw"})
        ], style={"display": "inline-block", "width": "80%"}),

        html.Div([
            html.Label("Select Layer:", style={"fontWeight": "bold", "textAlign": "center"}),
            dcc.Slider(
                id="map-slider",
                min=0, max=9, step=None, value=0,
                marks=marks,
                vertical=True,
                updatemode='drag',
                tooltip={"always_visible": False, "placement": "right"},
                className="custom-slider"
            ),
            html.Br(),
            dcc.Checklist(
                id="local",
                options=[{'label': 'Localize', 'value': 'checked'}],
                value=[],
                style={"margin-top": "20px", "textAlign": "center"}
            ),
        ], style={
            "display": "inline-block",
            "verticalAlign": "top",
            "width": "15%",
            "height": "85vh",
            "padding": "20px",
            "borderLeft": "1px solid #ccc"
        }),
    ], style={"display": "flex", "justifyContent": "center"}),

    dcc.Store(id="current_map", data="base")
])

# -----------------------
# Custom CSS (inline)
# -----------------------
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .rc-slider-rail { background-color: #ddd !important; }
            .rc-slider-track { background-color: transparent !important; }
            .rc-slider-dot {
                background-color: #aaa !important;
                border: 2px solid #888 !important;
            }
            .rc-slider-dot-active {
                background-color: #1E90FF !important;
                border-color: #1E90FF !important;
                box-shadow: 0 0 6px #1E90FF;
            }
            .rc-slider-handle {
                border-color: #1E90FF !important;
                background-color: #1E90FF !important;
                box-shadow: 0 0 6px #1E90FF;
            }
        </style>
        {%scripts%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# -----------------------
# Helper: generate wind arrows
# -----------------------
def generate_wind_traces(df, arrow_scale=.05, step=2):
    traces = []
    annotations = []
    df_sub = df.iloc[::step, :]

    # Normalize color by wind speed
    min_speed = df_sub["wind_speed_10m"].min()
    max_speed = df_sub["wind_speed_10m"].max()

    # Map-based scaling — all arrows have same geographic size
    lon_span = df["longitude"].max() - df["longitude"].min()
    lat_span = df["latitude"].max() - df["latitude"].min()
    arrow_len = min(lon_span, lat_span) * arrow_scale  # constant length for all

    for _, row in df_sub.iterrows():
        x0, y0 = row["longitude"], row["latitude"]
        direction = row["wind_direction_10m"]
        speed = row["wind_speed_10m"]

        # Wind direction (meteorological — from direction, so we invert)
        rad = np.radians(direction)
        x1 = x0 + arrow_len * np.sin(rad)
        y1 = y0 + arrow_len * np.cos(rad)

        # Color by speed
        color_frac = (speed - min_speed) / (max_speed - min_speed + 1e-6)
        color_rgb = f"rgb({int(30 + 200 * color_frac)}, {int(120 + 100 * color_frac)}, 255)"

        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor=color_rgb, opacity=0.9, showarrow=True
        ))

    # Dummy colorbar trace
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=[[0, "lightblue"], [1, "darkblue"]],
            cmin=min_speed, cmax=max_speed, color=[min_speed, max_speed],
            colorbar=dict(title="Wind Speed (m/s)"), size=0
        ),
        showlegend=False, hoverinfo="none"
    ))

    return traces, annotations

# -----------------------
# Base Terrain Layer
# -----------------------
def generate_terrain(df, local_checked):
    z_col = "elevation"
    df_clean = df.dropna(subset=["latitude", "longitude", z_col])
    lat_range = np.linspace(df_clean["latitude"].min(), df_clean["latitude"].max(), 200)
    lon_range = np.linspace(df_clean["longitude"].min(), df_clean["longitude"].max(), 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    z_grid = griddata(
        (df_clean["longitude"], df_clean["latitude"]),
        df_clean[z_col], (lon_grid, lat_grid), method="linear"
    )
    z_grid = np.nan_to_num(z_grid, nan=np.nanmean(df_clean[z_col]))
    zmin, zmax = (-500, 9000) if not local_checked else (np.nanmin(z_grid), np.nanmax(z_grid))

    return go.Contour(
        x=lon_range, y=lat_range, z=z_grid,
        colorscale="earth", zmin=zmin, zmax=zmax,
        contours=dict(showlines=False),
        colorbar=dict(title="Elevation (m)", x=1.02)
    )


# -----------------------
# Callback
# -----------------------
@app.callback(
    Output("map", "figure"),
    Output("current_map", "data"),
    Input("map-slider", "value"),
    Input("local", "value"),
    Input("current_map", "data")
)
def update_map(slider_value, local_value, current_map):
    current_map = map_options.get(slider_value, "base")
    local_checked = 'checked' in (local_value or [])

    # Always start with terrain base
    terrain_trace = generate_terrain(df, local_checked)
    fig = go.Figure(data=[terrain_trace])

    # Overlay selected layer
    if current_map == "wind_10m":
        traces, annotations = generate_wind_traces(df)
        for t in traces:
            fig.add_trace(t)
        fig.update_layout(annotations=annotations, title="Terrain + Wind Field")

    elif current_map in ["precipitation", "temperature", "humidity", "air_pressure"]:
        col_map = {
            "precipitation": "precipitation",
            "temperature": "temperature_2m",
            "humidity": "relative_humidity_2m",
            "air_pressure": "surface_pressure"
        }
        col = col_map[current_map]
        pivot = df.pivot_table(index="latitude", columns="longitude", values=col)
        x_coords, y_coords, z_data = pivot.columns.values, pivot.index.values, pivot.values

        global_ranges = {
            "precipitation": (0, 300),
            "temperature": (-50, 60),
            "humidity": (0, 100),
            "air_pressure": (870, 1080)
        }
        global_colors = {
            "precipitation": [[0.0, "lightgray"], [0.2, "lightblue"], [0.5, "blue"], [0.8, "purple"], [1.0, "navy"]],
            "temperature": [[0.0, "#313695"], [0.25, "#74add1"], [0.5, "#ffffbf"], [0.75, "#f46d43"], [1.0, "#a50026"]],
            "humidity": [[0.0, "#f7fcf5"], [0.25, "#c7e9c0"], [0.5, "#74c476"], [0.75, "#238b45"], [1.0, "#00441b"]],
            "air_pressure": [[0.0, "#f7fbff"], [0.25, "#deebf7"], [0.5, "#9ecae1"], [0.75, "#3182bd"], [1.0, "#08519c"]],
        }

        zmin, zmax = global_ranges[current_map]
        if local_checked:
            zmin, zmax = np.nanmin(z_data), np.nanmax(z_data)

        overlay = go.Heatmap(
            x=x_coords, y=y_coords, z=z_data,
            colorscale=global_colors[current_map],
            zmin=zmin, zmax=zmax,
            opacity=0.6,
            colorbar=dict(title=col.replace("_", " ").title(), x=1.07)
        )
        fig.add_trace(overlay)
        fig.update_layout(title=f"Terrain + {col.replace('_', ' ').title()}")

    else:
        fig.update_layout(title="Terrain Elevation")

    fig.update_layout(
        xaxis_title="Longitude (°)",
        yaxis_title="Latitude (°)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig, current_map


# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
