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
        
        # Track positions for visualization
        self.agent_positions = {}
        self.survivor_positions = []
        self.rescued_survivors = []
        self.previous_distances = {}  # Track distance to nearest survivor for each agent
        
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
            observations[agent] = self._get_observation(agent)
        
        infos = {agent: {'position': self.agent_positions[agent]} for agent in self.agents}
        infos = {agent: {'position': self.agent_positions[agent]} for agent in self.agents}
        return observations, infos
    
    def _get_min_distance(self, agent):
        """Get Manhattan distance to nearest unrescued survivor."""
        agent_pos = self.agent_positions[agent]
        
        # Get unrescued survivors
        unrescued = [s for s in self.survivor_positions if s not in self.rescued_survivors]
        
        if len(unrescued) == 0:
            return 0
        
        min_dist = float('inf')
        for survivor_pos in unrescued:
            dist = abs(agent_pos[0] - survivor_pos[0]) + abs(agent_pos[1] - survivor_pos[1])
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def _get_observation(self, agent):
        """Get meaningful observation for the agent.
        
        Returns a 4D observation:
        - normalized x position (0-1)
        - normalized y position (0-1)
        - normalized x distance to nearest survivor (-1 to 1)
        - normalized y distance to nearest survivor (-1 to 1)
        """
        agent_pos = self.agent_positions[agent]
        
        # Get unrescued survivors
        unrescued = [s for s in self.survivor_positions if s not in self.rescued_survivors]
        
        if len(unrescued) == 0:
            # No survivors left, return current position only
            obs = np.array([
                agent_pos[0] / self.grid_size,
                agent_pos[1] / self.grid_size,
                0.0,
                0.0
            ], dtype=np.float32)
        else:
            # Find nearest survivor
            min_dist = float('inf')
            nearest_survivor = None
            for survivor_pos in unrescued:
                dist = abs(agent_pos[0] - survivor_pos[0]) + abs(agent_pos[1] - survivor_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_survivor = survivor_pos
            
            # Calculate relative direction to nearest survivor
            dx = (nearest_survivor[0] - agent_pos[0]) / self.grid_size
            dy = (nearest_survivor[1] - agent_pos[1]) / self.grid_size
            
            obs = np.array([
                agent_pos[0] / self.grid_size,
                agent_pos[1] / self.grid_size,
                dx,
                dy
            ], dtype=np.float32)
        
        return obs
    
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
            reward = 0.0
            
            # Check if agent found a survivor
            agent_pos = self.agent_positions[agent]
            for survivor_pos in self.survivor_positions:
                if agent_pos == survivor_pos and survivor_pos not in self.rescued_survivors:
                    reward += 10.0  # Big reward for rescuing a survivor!
                    self.rescued_survivors.append(survivor_pos)
                    infos[agent] = infos.get(agent, {})
                    infos[agent]['rescued'] = True
            
            # Reward for getting closer to nearest survivor
            current_distance = self._get_min_distance(agent)
            previous_distance = self.previous_distances.get(agent, current_distance)
            
            if len(self.rescued_survivors) < len(self.survivor_positions):
                # Distance-based reward: positive for getting closer, negative for moving away
                distance_reward = (previous_distance - current_distance) * 0.1
                reward += distance_reward
            
            # Small penalty for each step to encourage efficiency
            reward -= 0.01
            
            # Update previous distance for next step
            self.previous_distances[agent] = current_distance
            
            observations[agent] = self._get_observation(agent)
            rewards[agent] = reward
            
            # Terminate when all survivors are rescued
            all_rescued = len(self.rescued_survivors) >= len(self.survivor_positions)
            terminations[agent] = all_rescued
            truncations[agent] = False
            infos[agent] = {'position': self.agent_positions[agent]}
        
        return observations, rewards, terminations, truncations, infos