import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from gymnasium import spaces


class Env(ParallelEnv):
    """
    First Rescue Env
    precondition: grid_size has to be greater than 2
    """
    
    def __init__(self, num_agents=3, grid_size=10):
        # Store parameters (don't use num_agents as it's a PettingZoo property)
        self._num_agents = num_agents
        self.grid_size = grid_size
        
        # Create map
        self.map = np.zeros((self.grid_size, self.grid_size), dtype = np.int32)
        
        # Agent names
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:] # copy
        
        # Simple observation and action spaces
        self.observation_space = spaces.Box(low=0, high=2, shape=(5,), dtype=np.int32) # curr pos, up, down, left, right
        self.action_space = spaces.Discrete(4)  #up, down, left, right
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment and Randomize map (location of surivivors, obstacles etc)

        Note:  0 - Moveable spot
               1 - Survivor
               2 - Agent
              -1 - out of bounds
        """
        self.agents = self.possible_agents[:]

        # Randomly spread the survivors throughout the map
        self.num_survivors = np.random.randint(low=1,high=self.grid_size, dtype = np.int32)
        for _ in range(self.num_survivors):
            x = np.random.randint(low=0,high=self.grid_size, dtype = np.int32)
            y = np.random.randint(low=0,high=self.grid_size, dtype = np.int32)
            self.map[x, y] = 1

        # Set positions to origin
        self.pos = {agent: [0,0] for agent in self.agents}

        # Hardcoded observations based on origin. 
        observations = {}
        for agent in self.agents:        
            observations[agent] = np.array([
                2,                   #curr_cell_type
                self.get_cell(0,-1), #up_cell_type
                self.get_cell(0,1),  #down_cell_type
                self.get_cell(-1,0), #left_cell_type
                self.get_cell(1,0)], #right_cell_type
                dtype=np.int32
                )
        
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
            self.pos[agent], penalty = self.update_positions(actions[agent], self.pos[agent])
            observations[agent] = self.get_observations(self.pos[agent])
            rewards[agent] =  self.get_rewards(self.pos[agent]) + penalty # Random reward
            terminations[agent] = True if self.num_survivors == 0 else False
            truncations[agent] = False
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos
    
    def get_cell(self,x,y):
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return -1
        return self.map[x, y]
    
    def get_rewards(self, pos):
        x,y = pos
        if self.get_cell(x,y) == 0: #on empty cell
            return -0.1
        
        elif self.get_cell(x,y) == 1: #on survivor
            self.map[x,y] = 0 #remove survivor
            return 1
        
        elif self.get_cell(x,y) == 2: #on another agent
            return -0.5 #discourage cluster and encourage exploration

    def get_observations(self, pos):
        x,y = pos 
        return np.array([
            self.get_cell(x,y),
            self.get_cell(x,y-1),
            self.get_cell(x,y+1),
            self.get_cell(x-1,y),
            self.get_cell(x+1,y)],
            dtype=np.int32
            )
    def update_positions(self, action, pos):
        """
        function will consider the actions for each agent and then update the positions accordingly

        up = 0
        down = 1
        left = 2
        right = 3

        """ 
        x,y = pos
        if action == 0 and y-1 >= 0:
            return [x, y-1], 0.0
        
        elif action == 1 and y+1 < self.grid_size:
            return [x, y+1], 0.0

        elif action == 2 and x-1 >= 0:
            return [x-1, y], 0.0

        elif action == 3 and x+1 < self.grid_size:
            return [x+1, y], 0.0
        
        else: # Penalize for invalid move
            return [x, y], -0.1