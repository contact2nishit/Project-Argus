import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim



# --- Setup the Open-Meteo API client ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- Configurable inputs ---
bbox = {
    "min_lat": 40.0,
    "max_lat": 55.0,
    "min_lon": -5.0,
    "max_lon": 15.0
}


# Days
# time_pre : [0, 1, 2, 3, 5, 7, 14, 31, 61, 92]
# time_post : [1, 3, 7, 14, 16]
time_pre = 3
time_post = 1

def bbox_to_region(bbox):
    """Return approximate country/region name for a bounding box."""
    # Compute centroid (center point of bbox)
    lat_center = (bbox["min_lat"] + bbox["max_lat"]) / 2
    lon_center = (bbox["min_lon"] + bbox["max_lon"]) / 2

    # Create a geolocator instance
    geolocator = Nominatim(user_agent="bbox_locator")

    # Query OpenStreetMap's Nominatim reverse geocoder
    location = geolocator.reverse((lat_center, lon_center), language="en")

    if location and "address" in location.raw:
        addr = location.raw["address"]
        # Pick out meaningful fields
        country = addr.get("country", "")
        state = addr.get("state", "")
        region = addr.get("region", "")
        # Combine the most relevant parts
        result = ", ".join([x for x in [region, state, country] if x])
        return result
    else:
        return "Unknown region"

region_name = bbox_to_region(bbox)
print(f"Region detected: {region_name}")

time_delta = time_post+time_pre

lat_delta = np.abs(np.sin(bbox["min_lat"])-np.sin(bbox["max_lat"]))
lon_delta = np.abs(bbox["min_lon"] - bbox["max_lon"])
sq_km = 6371*np.pi*lat_delta * lon_delta / 180

print(f"Retreiveing Weather Data Over {time_delta} days for {sq_km:.2f} km^2")

# --- Utility functions ---
def km_to_deg_lat(km):
    """Convert kilometers to degrees latitude (approximate)."""
    return km / 111.0

def km_to_deg_lon(km, lat):
    """Convert kilometers to degrees longitude (adjusted for latitude)."""
    return km / (111.320 * np.cos(np.radians(lat)))

def adaptive_resolution(bbox, time_pre, time_post):
    """Compute adaptive spatial resolution (1–100 km)."""
    lat_span = bbox["max_lat"] - bbox["min_lat"]
    lon_span = bbox["max_lon"] - bbox["min_lon"]
    area_factor = np.sqrt(lat_span * lon_span)
    time_span_days = time_pre + time_post
    km = 5 + (area_factor * 10) + (time_span_days * 5)
    return float(np.clip(km, 1, 100))

def generate_adaptive_grid(bbox, time_pre, time_post):
    """Generate coordinate grid within bbox using adaptive spacing."""
    km = adaptive_resolution(bbox, time_pre, time_post)
    # Step in degrees latitude
    step_lat_deg = km_to_deg_lat(km)
    # Use the midpoint latitude to approximate correct longitudinal step
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    step_lon_deg = km_to_deg_lon(km, mid_lat)

    lats = np.arange(bbox["min_lat"], bbox["max_lat"] + step_lat_deg, step_lat_deg)
    lons = np.arange(bbox["min_lon"], bbox["max_lon"] + step_lon_deg, step_lon_deg)
    coords = [(float(lat), float(lon)) for lat in lats for lon in lons]

    print(f"Adaptive step = {km:.1f} km (~{step_lat_deg:.3f}° lat, ~{step_lon_deg:.3f}° lon)")
    print(f"Grid size = {len(lats)}×{len(lons)} = {len(coords)} points\n")
    return coords, lats, lons

# --- Generate adaptive coordinate grid ---
coords, lats, lons = generate_adaptive_grid(bbox, time_pre, time_post)

# --- Open-Meteo request variables ---
url = "https://api.open-meteo.com/v1/forecast"
request_data = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m"
]

df_list = []

for lat, lon in coords:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": request_data,
        "forecast_days": time_post,
        "past_days": time_pre,
        "timeformat": "unixtime"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
    except Exception as e:
        print(f"Skipped {lat},{lon} due to API error: {e}")
        continue

    # Build the time index
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    # Extract hourly variables
    for i, name in enumerate(request_data):
        try:
            hourly_data[name] = hourly.Variables(i).ValuesAsNumpy()
        except Exception as e:
            print(f"Failed to read {name} at {lat},{lon}: {e}")
            hourly_data[name] = np.full(len(hourly_data["date"]), np.nan)

    df_t = pd.DataFrame(hourly_data)
    df_t["latitude"] = lat
    df_t["longitude"] = lon
    df_list.append(df_t)

# --- Combine all data ---
combined_df = pd.concat(df_list)
combined_df.set_index(["latitude", "longitude", "date"], inplace=True)

print("\nCombined DataFrame created successfully")
print(combined_df.head())
print(f"Total shape: {combined_df.shape}")

# --- Pressure interpolation step ---
print("\nInterpolating missing 'pressure_msl' values...")

# Reset index for interpolation
df_reset = combined_df.reset_index()

# Prepare output list
interp_frames = []

for timestamp, group in df_reset.groupby("date"):
    valid = group.dropna(subset=["pressure_msl"])
    if len(valid) < 3:
        # Not enough points for interpolation — skip
        interp_frames.append(group)
        continue

    # Define interpolation inputs
    points = valid[["latitude", "longitude"]].to_numpy()
    values = valid["pressure_msl"].to_numpy()

    # Interpolate for all points (linear interpolation in 2D)
    grid_points = group[["latitude", "longitude"]].to_numpy()
    interpolated = griddata(points, values, grid_points, method="linear")

    # Fill NaNs only where pressure was missing
    group["pressure_msl"] = np.where(
        group["pressure_msl"].isna(), interpolated, group["pressure_msl"]
    )

    interp_frames.append(group)

# Recombine
interpolated_df = pd.concat(interp_frames)
interpolated_df.set_index(["latitude", "longitude", "date"], inplace=True)

print(interpolated_df.head())


# -------------------
#  Temp code for viz
# --------------------
import datetime

# --- Save only the most recent timestamp ---
# Find the latest available datetime in your data
latest_time = interpolated_df.reset_index()["date"].max()

# Filter DataFrame for just that timestamp
df_current = interpolated_df.reset_index()
df_current = df_current[df_current["date"] == latest_time]

# --- Rename columns to your requested names ---
rename_map = {
    "longitude": "x",
    "latitude": "y",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "precipitation": "precipitation",
    "pressure_msl": "air_pressure"
}

# Apply renaming and handle missing 'phase'
df_current = df_current.rename(columns=rename_map)
df_current["phase"] = "N/A"  # Placeholder — could be "solid/liquid" if you calculate later

# --- Reorder columns ---
df_current = df_current[
    ["x", "y", "wind_speed", "wind_direction", "temperature",
     "humidity", "precipitation", "phase", "air_pressure"]
]

# --- Save to CSV ---
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"weather_snapshot_{timestamp_str}.csv"
df_current.to_csv(output_file, index=False)

print(f"Saved latest weather snapshot to '{output_file}'")
print(df_current.head())
