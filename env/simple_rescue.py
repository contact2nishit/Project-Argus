"""
Basic Rescue Environment

Simple multi-agent environment for rescue drone coordination.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pettingzoo import ParallelEnv
from gymnasium import spaces
from noise import pnoise2


class SimpleRescueEnv(ParallelEnv):
    """Simple rescue environment."""
    
    def __init__(self, num_agents=3, grid_size=10, render_mode = None, num_survivors = 1):
        # Store parameters (don't use num_agents as it's a PettingZoo property)
        self._num_agents = num_agents
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Agent names
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Environment State Information
        self.agent_positions = {}
        self.survivor_positions = {}
        self.num_survivors = num_survivors
        self.terrain_map = None

        # Rendering information
        self.fig = None
        self.ax = None
        self.agent_scatter = None
        self.survivor_scatter = None

        # Simple observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # up, down, left, right

    # Noise generation function

    def _generate_noise(self):
        """Generate Perlin noise for terrain."""

        # Terrain parameters
        width = self.grid_size
        height = self.grid_size
        scale = 100.0  # Controls the "zoom" level of the noise
        octaves = 6    # Number of layers of noise to combine
        persistence = 0.5 # How much each octave contributes to the overall shape
        lacunarity = 2.0 # How much the frequency increases with each octave
        base = int(np.random.uniform(0.0, 500.0)) # Seed for the noise generation

        for i in range(height):
            for j in range(width):
                # pnoise2 generates 2D Perlin noise
                self.terrain_map[i][j] = pnoise2(i / scale, j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=width,
                                        repeaty=height,
                                        base=base)

    # Environmental Housekeeping Functions

    def _build_observation(self, agent_id):
        """Build observation vector for a given agent."""

        obs_comps = []

        # Position vector [x, y]
        agent_pos = self.agent_positions[agent_id] / self.grid_size
        obs_comps.append(agent_pos.astype(np.float32))

        # Relative positions to agents
        for other_id in self.possible_agents:
            if other_id != agent_id:
                other_pos_norm = self.agent_positions[other_id] / self.grid_size
                rel_pos = other_pos_norm - agent_pos
                obs_comps.append(rel_pos.astype(np.float32))

        # Relative positions to survivors
        for survivor_pos in self.survivor_positions.values():
            survivor_pos_norm = survivor_pos / self.grid_size
            rel_pos = survivor_pos_norm - agent_pos
            obs_comps.append(rel_pos.astype(np.float32))

        # Reward function from environment state
        '''Not done yet'''


        return np.concatenate(obs_comps)
        
    def _terrain_builder(self, use_noise = False):
        """Generate a flat terrain map."""
        self.terrain_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if use_noise:
            self._generate_noise()

    def reset(self, use_noise = False, seed=None, options=None):
        """Reset the environment."""
        self.agents = self.possible_agents[:]

        # Generate terrain
        self._terrain_builder(use_noise)

        # Random positions for agents
        for agent in self.agents:
            self.agent_positions[agent] = np.random.uniform(0, self.grid_size, size=(2,))
        
        # Modular observation building
        observations = {}
        for agent in self.agents:
            observations[agent] = self._build_observation(agent)
        
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """Execute one step."""
        # Simple random rewards and termination
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Update agent positions based on actions
        for agent, action in actions.items():
            if action == 0:   # up
                self.agent_positions[agent][1] = min(self.grid_size, self.agent_positions[agent][1] + 1)
            elif action == 1: # down
                self.agent_positions[agent][1] = max(0, self.agent_positions[agent][1] - 1)
            elif action == 2: # left
                self.agent_positions[agent][0] = max(0, self.agent_positions[agent][0] - 1)
            elif action == 3: # right
                self.agent_positions[agent][0] = min(self.grid_size, self.agent_positions[agent][0] + 1)

        # Build observations, rewards, terminations
        for agent in self.agents:
            observations[agent] = self._build_observation(agent)
            rewards[agent] = np.random.random() - 0.5  # Random reward
            terminations[agent] = np.random.random() < 0.01  # 1% chance to terminate
            truncations[agent] = False
            infos[agent] = {}

        
        return observations, rewards, terminations, truncations, infos
    
    # Rendering functions

    def render(self):
        """Display the environment state."""

        # Skip rendering if not in human mode
        if self.render_mode != "human":
            return  

        # Initialize figure on first method call
        if self.fig is None or self.ax is None:
            plt.ion() # Enables interactive mode for matplotlib
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_title("Simple Rescue Environment")
            self.ax.grid()
            plt.imshow(self.terrain_map, cmap='terrain', origin='lower') # 'terrain' colormap for elevation
            plt.colorbar(label='Elevation')
            plt.title('Perlin Noise Generated Terrain')
            plt.show()
        
        # Clear previous plots
        if self.agent_scatter is not None:
            self.agent_scatter.remove()
        if self.survivor_scatter is not None:
            self.survivor_scatter.remove()

        # Plot agents
        if self.agent_positions:
            agent_pos = np.array([pos for pos in self.agent_positions.values()])
            self.agent_scatter = self.ax.scatter(agent_pos[:,0], agent_pos[:,1], c='blue', label='Agents')

        # Plot survivors 
        if self.survivor_positions:
            survivor_pos = np.array([pos for pos in self.survivor_positions.values()])
            self.survivor_scatter = self.ax.scatter(survivor_pos[:,0], survivor_pos[:,1], c='red', marker='x', label='Survivors')

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01) # Small pause to update the plot

    def close(self):
        """Close the rendering window."""
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None



