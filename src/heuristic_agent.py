"""
Heuristic Agent

Simple rule-based agent that uses basic heuristics on observations.
"""

import numpy as np
from src.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """Heuristic agent that follows simple rules based on observations.
    
    Strategy:
    - Interprets the 4 random observation values as rough directional hints
    - Uses simple thresholds to decide which direction to move
    - Acts deterministically based on observation patterns
    
    Note: Currently the environment provides random observations, so this
    agent will show how heuristics work, even if not optimal yet.
    """
    
    def __init__(self, agent_id, action_space):
        """Initialize the heuristic agent.
        
        Args:
            agent_id: Unique identifier for this agent
            action_space: The action space of the environment
        """
        super().__init__(agent_id)
        self.action_space = action_space
    
    def act(self, observation):
        """Choose action based on observation using heuristics.
        
        Current observation: [val1, val2, val3, val4] - 4 random values (0-1)
        
        Heuristic interpretation:
        - Compare val1 and val2 to decide vertical movement (up vs down)
        - Compare val3 and val4 to decide horizontal movement (left vs right)
        - Choose the direction with the strongest signal
        
        Action space: 0=up, 1=down, 2=left, 3=right
        
        Args:
            observation: Current observation from environment (4 values)
            
        Returns:
            action: Integer action (0-3)
        """
        val1, val2, val3, val4 = observation
        
        # Calculate vertical preference (up vs down)
        vertical_signal = val1 - val2  # Positive = prefer up, Negative = prefer down
        
        # Calculate horizontal preference (left vs right)
        horizontal_signal = val3 - val4  # Positive = prefer left, Negative = prefer right
        
        # Choose the axis with stronger signal
        if abs(vertical_signal) > abs(horizontal_signal):
            # Move vertically
            if vertical_signal > 0:
                return 0  # Move up
            else:
                return 1  # Move down
        else:
            # Move horizontally
            if horizontal_signal > 0:
                return 2  # Move left
            else:
                return 3  # Move right
    
    def learn(self, experience):
        """Heuristic agent doesn't learn.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        # Heuristic agent doesn't learn from experience
        pass
