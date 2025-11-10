"""
Random Agent - Simple baseline implementation
"""

import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects random actions."""
    
    def __init__(self, agent_id: str, action_space=4):
        super().__init__(agent_id)
        self.action_space = action_space
        
        
    
    def act(self, observation):
        """Select a random action."""
        return self.action_space.sample()
    
    def learn(self, experience):
        """Random agent doesn't learn."""
        pass