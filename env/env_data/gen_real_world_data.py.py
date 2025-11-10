import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import random
from tqdm import tqdm


# --- Configs ---
bbox = {
    "min_lat": 32.511, 
    "max_lat": 33.299, 
    "min_lon": -97.558,
    "max_lon": -96.292
}

date_start, date_end = "2025-01-01", "2025-01-02"
vars_hourly = [
    "temperature_2m", 
    "surface_pressure", 
    "cloud_cover_low",
    "precipitation",
    "relative_humidity_2m",
    "visibility",
    "wind_speed_10m", 
    "wind_direction_10m", 
    "wind_gusts_10m",
    "temperature_80m", 
    "wind_speed_80m", 
    "wind_direction_80m",
]

# --- get grid from bbox ---
def generate_grid(bbox, step_km=3):
    step_lat = step_km / 111.574
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    step_lon = step_km / (111.320 * np.cos(np.radians(mid_lat)))
    
    lats = np.arange(bbox["min_lat"], bbox["max_lat"] + step_lat, step_lat)
    lons = np.arange(bbox["min_lon"], bbox["max_lon"] + step_lon, step_lon)
    coords = [(float(lat), float(lon)) for lat in lats for lon in lons]
    print(f"Grid size: {len(lats)}x{len(lons)} = {len(coords)} points")
    return coords

# --- identify region (sanity check) ---
def bbox_to_region(bbox):
    lat_center = (bbox["min_lat"] + bbox["max_lat"]) / 2
    lon_center = (bbox["min_lon"] + bbox["max_lon"]) / 2
    geolocator = Nominatim(user_agent="bbox_locator", timeout=10)
    try:
        location = geolocator.reverse((lat_center, lon_center), language="en")
        if location and "address" in location.raw:
            addr = location.raw["address"]
            return ", ".join(filter(None, [addr.get("region",""), addr.get("state",""), addr.get("country","")]))
    except:
        return "Unknown region"
    return "Unknown region"

# --- Async limits ---
SEM = asyncio.Semaphore(5)
REQUEST_DELAY = 0.5 # Prevents Free API call limit
MAX_RETRIES = 3
MAX_FAILURE_RATE = 0.05  # 5% failure tolerance

# --- Async fetch ---
async def fetch_weather(session, lat, lon):
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": vars_hourly,
        "start_date": date_start,
        "end_date": date_end,
        "timeformat": "unixtime"
    }
    
    for attempt in range(MAX_RETRIES):
        async with SEM:
            try:
                async with session.get(url, params=params, timeout=20) as resp:
                    data = await resp.json()
            except Exception as e:
                print(f"Skipped {lat},{lon} due to API error: {e}")
                data = {"error": True, "reason": str(e)}

            # --- Add pacing delay ---
            await asyncio.sleep(REQUEST_DELAY)

        if not data.get("error"):
            break
        else:
            wait = 3 * random.random()  # 0-3 s delay gives time for API call to "cool down"
            print(f"Retry {attempt+1} for {lat},{lon} after {wait:.2f}s due to {data.get('reason')}")
            await asyncio.sleep(wait)
    else:
        return None  # Failed after retries

    try:
        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"], unit="s", utc=True)
        gts_df = pd.DataFrame({name: hourly.get(name, [np.nan]*len(times)) for name in vars_hourly})
        gts_df["latitude"] = lat
        gts_df["longitude"] = lon
        gts_df["elevation"] = data.get("elevation", np.nan)
        gts_df["date"] = times
        return gts_df
    except Exception as e:
        print(f"Failed to parse {lat},{lon}: {e}")
        return None


async def main():
    coords = generate_grid(bbox)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_weather(session, lat, lon) for lat, lon in coords]

        results = []
        failed = 0
        total = len(tasks)

        for f in tqdm(asyncio.as_completed(tasks), total=total, desc="Fetching Environment data"):
            res = await f
            if res is None:
                failed += 1
            results.append(res)

            # --- Check failure rate dynamically ---
            if failed / total > MAX_FAILURE_RATE:
                raise RuntimeError(
                    f"Too many failed fetches ({failed}/{total}, >{MAX_FAILURE_RATE*100:.0f}% failure rate). Aborting."
                )

    # Combine dataframes
    gsts_df = [r for r in results if r is not None]
    if not gsts_df:
        print("No data fetched!")
        return

    gsts_df = pd.concat(gsts_df)
    gsts_df.set_index(["latitude", "longitude", "date"], inplace=True)
    print(f"Combined DataFrame: {gsts_df.shape}")

    os.makedirs("env/env_data", exist_ok=True)
    output_path = "env/env_data/env_snapshot.csv"
    gsts_df.to_csv(output_path)
    print(f"Saved weather snapshot: {output_path}")


if __name__ == "__main__":
    print(f"Region detected: {bbox_to_region(bbox)}")
    asyncio.run(main())
