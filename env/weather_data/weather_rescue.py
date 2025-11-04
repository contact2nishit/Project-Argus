import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pettingzoo import ParallelEnv
from gymnasium import spaces

class WeatherRescueEnv(ParallelEnv):
    """Rescue environment with weather effects and dynamic visualization."""
    
    def __init__(self, num_agents=3, grid_size=10, weather_path="env\\weather_data"):
        self._num_agents = num_agents
        self.grid_size = grid_size
        self.weather_path = weather_path
        
        # Agent names
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # Observation & action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)  # 8 directions
        
        # Track positions
        self.agent_positions = {}
        self.survivor_positions = []
        
        # Weather grids
        self.wind_speed = np.zeros((grid_size, grid_size))
        self.wind_dir = np.zeros((grid_size, grid_size))  # degrees
        self.rain = np.zeros((grid_size, grid_size))
        
        self.load_weather_data()
        
        # Map actions to (dx, dy)
        self.action_map = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1),    # right
            4: (-1, -1),  # up-left
            5: (-1, 1),   # up-right
            6: (1, -1),   # down-left
            7: (1, 1),    # down-right
        }

        # Matplotlib figure
        self.fig, self.ax = None, None
        self.quiver = None
        self.im = None
        self.scatter = None

    def load_weather_data(self):
        """Load CSV weather data into numpy arrays."""
        wind_file = os.path.join(self.weather_path, "wind.csv")
        rain_file = os.path.join(self.weather_path, "rain.csv")
        
        # Wind
        with open(wind_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x, y = int(row['x']), int(row['y'])
                self.wind_speed[x, y] = float(row['wind_speed'])
                self.wind_dir[x, y] = float(row['wind_direction'])
        
        # Rain
        with open(rain_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x, y = int(row['x']), int(row['y'])
                self.rain[x, y] = float(row['rainfall'])

    def reset(self, seed=None, options=None):
        """Reset environment."""
        self.agents = self.possible_agents[:]
        self.agent_positions = {agent: [np.random.randint(0, self.grid_size),
                                        np.random.randint(0, self.grid_size)]
                                for agent in self.agents}
        
        # Place 2-3 survivors randomly
        num_survivors = np.random.randint(2, 4)
        self.survivor_positions = [[np.random.randint(0, self.grid_size),
                                    np.random.randint(0, self.grid_size)]
                                   for _ in range(num_survivors)]
        
        # Observations
        observations = {agent: np.random.random(4).astype(np.float32) for agent in self.agents}
        infos = {agent: {'position': self.agent_positions[agent]} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """Step environment and apply wind effects."""
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent in self.agents:
            if agent in actions:
                dx, dy = self.action_map[actions[agent]]
                x, y = self.agent_positions[agent]
                
                # Apply wind effect
                wind_speed = self.wind_speed[x, y]
                wind_dir = np.radians(self.wind_dir[x, y])
                
                # Movement vector
                move_vector = np.array([dx, dy])
                wind_vector = np.array([np.sin(wind_dir), -np.cos(wind_dir)])  # N=0 deg
                penalty = np.dot(move_vector, wind_vector) * wind_speed
                
                dx_adj = dx - int(np.sign(penalty) * min(abs(penalty), 1))
                dy_adj = dy - int(np.sign(penalty) * min(abs(penalty), 1))
                
                # Update position
                new_x = max(0, min(self.grid_size-1, x + dx_adj))
                new_y = max(0, min(self.grid_size-1, y + dy_adj))
                self.agent_positions[agent] = [new_x, new_y]
            
            observations[agent] = np.random.random(4).astype(np.float32)
            
            rewards[agent] = np.random.random() - 0.5
            for sx, sy in self.survivor_positions:
                if self.agent_positions[agent] == [sx, sy]:
                    rewards[agent] += 1.0
            
            terminations[agent] = np.random.random() < 0.01
            truncations[agent] = False
            infos[agent] = {'position': self.agent_positions[agent]}
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Dynamic matplotlib render showing wind, rain, drones, and survivors."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            plt.ion()
            self.ax.set_xlim(-0.5, self.grid_size-0.5)
            self.ax.set_ylim(-0.5, self.grid_size-0.5)
            self.ax.set_xticks(range(self.grid_size))
            self.ax.set_yticks(range(self.grid_size))
            self.ax.grid(True)
        
        self.ax.clear()
        
        # Draw rain as heatmap
        self.ax.imshow(self.rain.T, origin='lower', cmap='Blues', alpha=0.5, interpolation='nearest')
        
        # Draw wind arrows
        X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        U = np.sin(np.radians(self.wind_dir))
        V = -np.cos(np.radians(self.wind_dir))
        self.ax.quiver(X, Y, U, V, self.wind_speed, scale=5, cmap='autumn')
        
        # Draw survivors
        sx, sy = zip(*self.survivor_positions)
        self.ax.scatter(sx, sy, marker='*', s=200, c='gold', label='Survivors')
        
        # Draw drones
        dx, dy = zip(*[self.agent_positions[a] for a in self.agents])
        self.ax.scatter(dx, dy, marker='o', s=100, c='red', label='Drones')
        
        self.ax.set_title("Weather Rescue Map")
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.legend(loc='upper right')
        plt.pause(0.1)
