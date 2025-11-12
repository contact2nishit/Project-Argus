import os
import numpy as np
import pandas as pd

# -----------------------
# CSV File
# -----------------------
# DATA_CSV = os.path.join("env", "weather_data", "static_weather_data.csv")
# df = pd.read_csv(DATA_CSV)

# ----------------
# Calc Force
# ----------------

def calculate_drone_forces(
    x, y,
    wind_speed, wind_direction,  # m/s, radians
    temperature, humidity,       # °C, 0-1
    precipitation, phase,        # mm/h, "rain"/"snow"/"none"
    air_pressure,                # Pa
    drone_mass,                  # kg
    rotor_count,                 # e.g., 4
    rotor_diameter,              # m
    rotor_speed_rpm,             # rev/min
    C_T=0.1,                     # thrust coefficient (typical)
    C_D=1.0,                      # drag coefficient
    drone_area=0.1               # m^2 frontal area
):
    g = 9.81  # gravity
    
    # 1. Air density (kg/m^3)
    T_k = temperature + 273.15
    rho = air_pressure * (1 - 0.378 * humidity) / (287 * T_k)
    
    # 2. Wind vector (m/s)
    v_wind = np.array([
        wind_speed * np.cos(wind_direction),
        wind_speed * np.sin(wind_direction),
        0.0  # horizontal wind
    ])
    
    # Assume drone at hover (v_drone = 0)
    v_rel = -v_wind
    F_drag = 0.5 * rho * C_D * drone_area * np.linalg.norm(v_rel) * v_rel
    
    # 3. Gravity force
    F_gravity = np.array([0, 0, drone_mass * g])
    
    # 4. Rotor thrust per rotor
    n = rotor_speed_rpm / 60  # rev/s
    T_per_rotor = C_T * rho * (n**2) * (rotor_diameter**4)
    
    # Thrust vector points upwards (+z)
    F_thrust = np.array([0, 0, rotor_count * T_per_rotor])
    
    # 5. Precipitation effect
    if phase.lower() == "rain":
        F_precip = np.array([0, 0, -0.01 * precipitation * drone_area])  # small downward effect
    elif phase.lower() == "snow":
        F_precip = np.array([0, 0, -0.005 * precipitation * drone_area])
    else:
        F_precip = np.array([0, 0, 0])
    
    # 6. Net force
    F_env = F_gravity + F_drag + F_precip
    F_net = F_thrust - F_env
    
    return {
        "air_density": rho,
        "F_drag": F_drag,
        "F_gravity": F_gravity,
        "F_precip": F_precip,
        "F_thrust": F_thrust,
        "F_env": F_env,
        "F_net": F_net
    }

# Example usage
forces = calculate_drone_forces(
    x=0, y=0,
    wind_speed=5, wind_direction=np.pi/4,
    temperature=20, humidity=0.5,
    precipitation=2, phase="rain",
    air_pressure=101325,
    drone_mass=1.5, rotor_count=4,
    rotor_diameter=0.3, rotor_speed_rpm=5000
)

print("Net force on drone:", forces["F_net"])

