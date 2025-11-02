"""
First Agent
"""

import numpy as np
from .base_agent import BaseAgent


class Agent(BaseAgent):
    """Agent that selects random actions."""
    
    def __init__(self, agent_id: str, action_space):
        super().__init__(agent_id)
        self.action_space = action_space
    
    def act(self, observation):
        """Select a random action."""
        return self.action_space.sample()
    
    def learn(self, experience):
        """Random agent doesn't learn."""
        pass