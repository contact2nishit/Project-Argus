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
        
        # Track positions for visualization
        self.agent_positions = {}
        self.survivor_positions = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.agents = self.possible_agents[:]
        
        # Initialize random positions for agents
        self.agent_positions = {}
        for agent in self.agents:
            self.agent_positions[agent] = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ]
        
        # Place 2-3 survivors randomly
        num_survivors = np.random.randint(2, 4)
        self.survivor_positions = []
        for _ in range(num_survivors):
            self.survivor_positions.append([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
        
        # Simple random observations for all agents
        observations = {}
        for agent in self.agents:
            observations[agent] = np.random.random(4).astype(np.float32)
        
        infos = {agent: {'position': self.agent_positions[agent]} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        """Execute one step."""
        # Action map: 0=up, 1=down, 2=left, 3=right
        action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        # Move agents based on actions
        for agent in self.agents:
            if agent in actions:
                action = actions[agent]
                delta = action_map[action]
                new_pos = [
                    self.agent_positions[agent][0] + delta[0],
                    self.agent_positions[agent][1] + delta[1]
                ]
                # Keep within bounds
                new_pos[0] = max(0, min(self.grid_size - 1, new_pos[0]))
                new_pos[1] = max(0, min(self.grid_size - 1, new_pos[1]))
                self.agent_positions[agent] = new_pos
        
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
            infos[agent] = {'position': self.agent_positions[agent]}
        
        return observations, rewards, terminations, truncations, infos