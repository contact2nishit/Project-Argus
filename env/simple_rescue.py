import numpy as np
import pandas as pd
from pettingzoo import ParallelEnv
from gymnasium import spaces

def calculate_drone_forces(
    x, y,
    wind_speed, wind_direction,
    temperature, humidity,
    precipitation, phase,
    air_pressure,
    drone_mass,
    rotor_count,
    rotor_diameter,
    rotor_speed_rpm,
    C_T=0.1,
    C_D=1.0,
    drone_area=0.1
):
    g = 9.81
    T_k = temperature + 273.15
    rho = air_pressure * (1 - 0.378 * humidity) / (287 * T_k)
    v_wind = np.array([
        wind_speed * np.cos(wind_direction),
        wind_speed * np.sin(wind_direction),
        0.0
    ])
    v_rel = -v_wind
    F_drag = 0.5 * rho * C_D * drone_area * np.linalg.norm(v_rel) * v_rel
    F_gravity = np.array([0, 0, drone_mass * g])
    n = rotor_speed_rpm / 60
    T_per_rotor = C_T * rho * (n**2) * (rotor_diameter**4)
    F_thrust = np.array([0, 0, rotor_count * T_per_rotor])
    # if phase.lower() == "rain":
    #     F_precip = np.array([0, 0, -0.01 * precipitation * drone_area])
    # elif phase.lower() == "snow":
    #     F_precip = np.array([0, 0, -0.005 * precipitation * drone_area])
    # else:
    F_precip = np.array([0, 0, 0])
    F_env = F_gravity + F_drag + F_precip
    F_net = F_thrust - F_env
    return {"F_net": F_net}

class CSVRescueEnv(ParallelEnv):
    """Rescue environment using CSV coordinates and drone physics."""

    def __init__(self, csv_file, weather_df, num_agents=3):
        self.df = pd.read_csv(csv_file)  # lat/lon grid
        if not {"lat","lon"}.issubset(self.df.columns):
            raise ValueError("CSV must have 'lat' and 'lon' columns")

        self.weather_df = weather_df.set_index(["lat","lon"])  # weather variables for each grid
        self.grid_points = self.df[["lat","lon"]].to_numpy()
        self.num_points = len(self.grid_points)
        self.index_to_coord = {i: tuple(self.grid_points[i]) for i in range(self.num_points)}

        self._num_agents = num_agents
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)  # placeholder, movement computed via forces
        self.agent_positions = {}
        self.survivor_positions = []

        # Drone parameters
        self.drone_mass = 1.5
        self.rotor_count = 4
        self.rotor_diameter = 0.3
        self.rotor_speed_rpm = 5000

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_positions = {agent: np.random.randint(0, self.num_points) for agent in self.agents}
        num_survivors = np.random.randint(2,4)
        self.survivor_positions = [np.random.randint(0, self.num_points) for _ in range(num_survivors)]
        observations = {agent: np.random.random(4).astype(np.float32) for agent in self.agents}
        infos = {agent: {'position': self.index_to_coord[self.agent_positions[agent]]} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            idx = self.agent_positions[agent]
            lat, lon = self.index_to_coord[idx]

            # Get nearest weather data
            try:
                w = self.weather_df.loc[(lat, lon)]
                wind_speed = w.wind_speed
                wind_dir = np.radians(w.wind_direction)
                temp = w.temperature
                hum = w.humidity
                prec = w.precipitation
                phase = w.phase
                airp = w.air_pressure
            except:
                wind_speed = wind_dir = temp = hum = prec = airp = 0
                phase = "none"

            # Compute net force
            forces = calculate_drone_forces(
                x=lat, y=lon,
                wind_speed=wind_speed, wind_direction=wind_dir,
                temperature=temp, humidity=hum,
                precipitation=prec, phase=phase,
                air_pressure=airp,
                drone_mass=self.drone_mass,
                rotor_count=self.rotor_count,
                rotor_diameter=self.rotor_diameter,
                rotor_speed_rpm=self.rotor_speed_rpm
            )

            F_net_xy = forces["F_net"][:2]  # horizontal net force

            # Move to nearest neighbor in direction of net force
            distances = self.grid_points - np.array([lat, lon])
            angles = np.arctan2(distances[:,1], distances[:,0])
            diffs = np.abs(angles - np.arctan2(F_net_xy[1], F_net_xy[0]))
            next_idx = np.argmin(diffs + np.linalg.norm(distances, axis=1)*0.1)  # small weight on distance
            self.agent_positions[agent] = next_idx

            # Random observations & rewards
            observations[agent] = np.random.random(4).astype(np.float32)
            rewards[agent] = np.random.random() - 0.5
            terminations[agent] = np.random.random() < 0.01
            truncations[agent] = False
            infos[agent] = {'position': self.index_to_coord[self.agent_positions[agent]]}

        return observations, rewards, terminations, truncations, infos
