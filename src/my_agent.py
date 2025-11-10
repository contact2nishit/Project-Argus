import numpy as np
from base_agent import BaseAgent
from calculate_forces import calculate_drone_forces

class SimpleDroneAgent(BaseAgent):
    """
    A simple agent that takes environmental observations and outputs an action.
    
    Action format:
        return a dict {
            "target_rotor_speed_rpm": float,
            "thrust_adjustment": float
        }
    
    This agent uses a heuristic: increase thrust if net Z-force < 0,
    reduce thrust if too high.
    """

    def __init__(self, agent_id: str, base_rotor_speed_rpm: float = 5000.0):
        super().__init__(agent_id)
        self.base_rotor_speed_rpm = base_rotor_speed_rpm

    def act(self, observation):
        """
        observation is expected to be a dictionary containing:
            - x, y
            - wind_speed, wind_direction
            - temperature, humidity
            - precipitation, phase
            - air_pressure
            - drone_mass, rotor_count, rotor_diameter
        """
        # Compute forces under current hypothetical rotor speed
        forces = calculate_drone_forces(
            x=observation["x"],
            y=observation["y"],
            wind_speed=observation["wind_speed"],
            wind_direction=observation["wind_direction"],
            temperature=observation["temperature"],
            humidity=observation["humidity"],
            precipitation=observation["precipitation"],
            phase=observation["phase"],
            air_pressure=observation["air_pressure"],
            drone_mass=observation["drone_mass"],
            rotor_count=observation["rotor_count"],
            rotor_diameter=observation["rotor_diameter"],
            rotor_speed_rpm=self.base_rotor_speed_rpm
        )

        F_net = forces["F_net"]
        net_vertical = F_net[2]

        # Heuristic: if net vertical force < 0, add thrust
        adjustment = 0.0
        if net_vertical < 0:
            adjustment = min(1000, abs(net_vertical) * 10)
        else:
            adjustment = -min(1000, net_vertical * 10)

        return {
            "target_rotor_speed_rpm": float(self.base_rotor_speed_rpm + adjustment),
            "thrust_adjustment": float(adjustment)
        }

    def learn(self, experience):
        """No learning yet, placeholder."""
        pass
