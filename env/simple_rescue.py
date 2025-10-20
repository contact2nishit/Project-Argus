"""
Basic Rescue Environment

Simple multi-agent environment for rescue drone coordination.
"""

import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from gymnasium import spaces


class SimpleRescueEnv(ParallelEnv):
    """Simple rescue environment."""
    
    def __init__(self, num_agents=3, grid_size=10):
        # Store parameters (don't use num_agents as it's a PettingZoo property)
        self._num_agents = num_agents
        self.grid_size = grid_size
        
        # Agent names
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # Simple observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.agents = self.possible_agents[:]
        
        # Simple random observations for all agents
        observations = {}
        for agent in self.agents:
            observations[agent] = np.random.random(4).astype(np.float32)
        
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
        
        for agent in self.agents:
            observations[agent] = np.random.random(4).astype(np.float32)
            rewards[agent] = np.random.random() - 0.5  # Random reward
            terminations[agent] = np.random.random() < 0.01  # 1% chance to terminate
            truncations[agent] = False
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos