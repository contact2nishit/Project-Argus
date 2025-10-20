"""
Base Agent Interface for Project Argus

Simple abstract base class for all RL agents.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    @abstractmethod
    def act(self, observation):
        """Select an action given an observation."""
        pass
    
    @abstractmethod
    def learn(self, experience):
        """Update the agent's policy."""
        pass