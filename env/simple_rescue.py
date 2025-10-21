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
        
        #make positions for each agent as a dictionary
        self.positions = {}
        for agent in self.agents:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0,self.grid_size)

            self.positions[agent] = [x,y]
        

        #make a position for only one survivor (eventually test with multiple different survivors) -> for loop
        self.survivor = [np.random.randint(0,self.grid_size), np.random.randint(0,self.grid_size)]




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

            ##include a if statement to see if the position of the agent is equal to the position of the survivor
            if np.array_equal(self.positions[agent], self.survivor):
                rewards[agent] = 1
                terminations[agent] = True
            ##if it is not the survivor
            else:
                rewards[agent] = -0.1
                terminations[agent] = False
        



            observations[agent] = np.random.random(4).astype(np.float32)
            truncations[agent] = False
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos