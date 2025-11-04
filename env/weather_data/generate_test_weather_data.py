# -----------------------------
# Weather Field Parameters for Dummy Drone Simulation
# -----------------------------
# 2D Static Flat
# Accounts for: wind, temperature, humidity, precipitation, air pressure
# DOES NOT account for: magnetic interference, mountains, buildings

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from env_presets import ENV_PRESETS

# -----------------------------
# Adjustable parameters
# -----------------------------
grid_shape = (50, 60)  # (rows, cols)

# -----------------------------
# Select Environment
# -----------------------------
env = "sporadic_storm"  # Choose from ENV_PRESETS: hot_desert, cold_tundra, stormy_area, etc.
params = ENV_PRESETS[env]

# -----------------------------
# Apply Environment Parameters
# -----------------------------

# Wind parameters
base_wind_speed     = params["base_wind_speed"]       # m/s, average wind speed
gust_amplitude      = params["gust_amplitude"]        # m/s, variation around base speed
base_wind_direction = params["base_wind_direction"]   # degrees, 0=north, 90=east
wind_shear          = params["wind_shear"]           # small turbulence coefficient, m/s
wind_variation      = params["wind_variation"]       # degrees, max random deviation from base direction

# Temperature parameters
base_temp           = params["base_temp"]           # °C, average temperature
temp_variation      = params["temp_variation"]      # °C, random variation around base_temp

# Humidity parameters
base_humidity       = params["base_humidity"]       # fraction (0-1), average relative humidity
humidity_variation  = params["humidity_variation"]  # fraction (0-1), random variation

# Precipitation parameters
base_precip         = params["base_precip"]         # mm/h or mm/day, average precipitation
precip_variation    = params["precip_variation"]    # mm/h or mm/day, random variation
snow_temp_threshold = params.get("snow_temp_threshold", 0)  # °C below which precipitation is snow

# Air pressure (Pa)
base_air_pressure = 101325  # Pa, standard atmospheric pressure at sea level

# -----------------------------
# Helper functions
# -----------------------------
def smooth_field(field, sigma=2):
    return gaussian_filter(field, sigma=sigma)

def smooth_direction_field(dir_field, sigma=2):
    """
    Smooth a wind direction field accounting for circular wrap-around (0-360°).
    """
    rad = np.radians(dir_field)
    x = np.cos(rad)
    y = np.sin(rad)
    x_smooth = gaussian_filter(x, sigma=sigma)
    y_smooth = gaussian_filter(y, sigma=sigma)
    smoothed_dir = np.degrees(np.arctan2(y_smooth, x_smooth)) % 360
    return smoothed_dir

def generate_wind_field(grid_shape, base_speed, gust_amp, base_dir, wind_variation=45, temperature=None, humidity=None, shear=0.0):
    rows, cols = grid_shape

    # Wind speed with gusts
    wind_speed = base_speed + np.random.uniform(-gust_amp*0.3, gust_amp*0.3, (rows, cols))
    if humidity is not None:
        wind_speed *= (1 - 0.3 * humidity)
    if temperature is not None:
        wind_speed += np.where(temperature < 0, 0.5, 0.0)
    wind_speed += np.random.uniform(-shear, shear, (rows, cols))
    wind_speed = np.clip(wind_speed, 0, None)

    # Random wind directions
    wind_dir = base_dir + np.random.uniform(-wind_variation, wind_variation, (rows, cols))

    # Smooth
    wind_speed = gaussian_filter(wind_speed, sigma=1)
    wind_dir = smooth_direction_field(wind_dir, sigma=2)

    return wind_speed, wind_dir

def generate_temperature_field(grid_shape, base_temp, temp_var, wind_speed=None, precip=None):
    rows, cols = grid_shape
    temp = base_temp + np.random.uniform(-temp_var, temp_var, (rows, cols))
    if precip is not None:
        temp -= 0.1 * precip
    if wind_speed is not None:
        temp_shift = (np.roll(temp, 1, axis=0) + np.roll(temp, -1, axis=0) +
                      np.roll(temp, 1, axis=1) + np.roll(temp, -1, axis=1)) / 4
        temp = 0.7 * temp + 0.3 * temp_shift * (wind_speed / np.max(wind_speed))
    temp = np.clip(temp, -50, 50)
    return smooth_field(temp, sigma=2)

def generate_humidity_field(grid_shape, base_hum, hum_var, wind_speed=None, temp=None):
    rows, cols = grid_shape
    hum = base_hum + np.random.uniform(-hum_var*0.5, hum_var*0.5, (rows, cols))
    if temp is not None:
        hum -= 0.05 * (temp - np.mean(temp))
    if wind_speed is not None:
        hum = 0.8 * hum + 0.2 * np.roll(hum, 1, axis=0)
    hum = np.clip(hum, 0, 1)
    return smooth_field(hum, sigma=2)

def generate_precipitation_field(grid_shape, base_precip, precip_var, humidity=None, temp=None):
    rows, cols = grid_shape
    precip = base_precip + np.random.uniform(0, precip_var*0.5, (rows, cols))
    if humidity is not None:
        precip += 10 * np.clip(humidity - 0.7, 0, 1)
    precip = np.clip(precip, 0, None)
    return smooth_field(precip, sigma=2)

def generate_air_pressure_field(grid_shape, base_pressure, temp=None, wind_speed=None):
    rows, cols = grid_shape
    pressure = base_pressure + np.random.uniform(-500, 500, (rows, cols))  # +-500 Pa variation
    if temp is not None:
        temp_K = temp + 273.15
        pressure *= temp_K / 288.15  # scale by temperature relative to 15°C
    if wind_speed is not None:
        pressure *= 1 - 0.01 * (wind_speed / np.max(wind_speed))
    pressure = np.clip(pressure, 90000, 110000)
    return smooth_field(pressure, sigma=1)

# -----------------------------
# Iterative interaction
# -----------------------------
def generate_interacting_weather_field(grid_shape, iterations=10, tolerance=0.01):
    wind_speed, wind_dir = generate_wind_field(grid_shape, base_wind_speed, gust_amplitude, base_wind_direction)
    temp = generate_temperature_field(grid_shape, base_temp, temp_variation)
    humidity = generate_humidity_field(grid_shape, base_humidity, humidity_variation)
    precip = generate_precipitation_field(grid_shape, base_precip, precip_variation)
    air_pressure = generate_air_pressure_field(grid_shape, base_air_pressure)

    for _ in range(iterations):
        prev_temp = temp.copy()
        prev_hum = humidity.copy()
        prev_precip = precip.copy()
        prev_air_pressure = air_pressure.copy()
        prev_wind = wind_speed.copy()

        wind_speed, wind_dir = generate_wind_field(
            grid_shape, base_wind_speed, gust_amplitude, base_wind_direction,
            wind_variation=wind_variation,
            temperature=temp, humidity=humidity, shear=wind_shear
        )
        temp = generate_temperature_field(grid_shape, base_temp, temp_variation, wind_speed=wind_speed, precip=precip)
        humidity = generate_humidity_field(grid_shape, base_humidity, humidity_variation, wind_speed=wind_speed, temp=temp)
        precip = generate_precipitation_field(grid_shape, base_precip, precip_variation, humidity=humidity, temp=temp)
        air_pressure = generate_air_pressure_field(grid_shape, base_air_pressure, temp=temp, wind_speed=wind_speed)

        max_change = max(
            np.max(np.abs(temp - prev_temp)),
            np.max(np.abs(humidity - prev_hum)),
            np.max(np.abs(precip - prev_precip)),
            np.max(np.abs(air_pressure - prev_air_pressure)),
            np.max(np.abs(wind_speed - prev_wind))
        )
        if max_change < tolerance:
            break

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_dir,
        "temperature": temp,
        "humidity": humidity,
        "precipitation": precip,
        "air_pressure": air_pressure
    }

# -----------------------------
# Main: Generate CSV
# -----------------------------
if __name__ == "__main__":
    weather_field = generate_interacting_weather_field(grid_shape)

    rows = []
    rows_count, cols_count = grid_shape
    for x in range(rows_count):
        for y in range(cols_count):
            temp_val = weather_field["temperature"][x, y]
            rows.append({
                "x": x,
                "y": y,
                "wind_speed": weather_field["wind_speed"][x, y],
                "wind_direction": weather_field["wind_direction"][x, y],
                "temperature": temp_val,
                "humidity": weather_field["humidity"][x, y],
                "precipitation": weather_field["precipitation"][x, y],
                "phase": "snow" if temp_val < snow_temp_threshold else "rain",
                "air_pressure": weather_field["air_pressure"][x, y]
            })

    df = pd.DataFrame(rows)
    df.to_csv("env\\weather_data\\static_weather_data.csv", index=False)
    print("Static weather saved to static_weather_data.csv")
