"""
Demo script for Project Argus

Includes a weather-based rescue simulation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.random_agent import RandomAgent
from env.simple_rescue import SimpleRescueEnv  
# from env.weather_data.weather_rescue import WeatherRescueEnv  # Not yet implemented  


def run_simple_demo():
    """Run a basic demo with visualization."""
    print("🚁 Project Argus - Visual Demo 🚁\n")
    print("Watch the drones search for survivors!\n")
    
    # Create environment
    env = SimpleRescueEnv(num_agents=3, grid_size=8)
    
    # Create random agents
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = RandomAgent(agent_id, env.action_space)
    
    # Run one episode
    observations, infos = env.reset()
    
    # Show initial state
    print(f"🎯 Mission: Find {len(env.survivor_positions)} survivors!")
    
    for step in range(20):  # Run 20 steps
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            if agent_id in observations:
                actions[agent_id] = agent.act(observations[agent_id])
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Show rewards
        total_reward = sum(rewards.values())
        print(f"💰 Total Reward: {total_reward:+.2f}")
        
        # Show drone positions
        for agent_id in env.agents:
            pos = infos[agent_id]['position']
            print(f"   {agent_id}: ({pos[0]}, {pos[1]})")
        
        
        # Check if done
        if any(terminations.values()) or any(truncations.values()):
            print("\n✅ Mission Complete!")
            break
    
    print("\n🎉 Demo completed!")
    print("=" * 40)


def run_weather_demo():
    """Run the weather-affected rescue demo."""
    print("🌤️ Project Argus - Weather Rescue Demo 🌤️\n")
    
    # Create environment
    # MAKE SURE GRID SIZE MATCHES DATA SPEC
    env = WeatherRescueEnv(num_agents=3, grid_size=50, weather_path="env\\weather_data")
    
    # Create random agents
    agents = {agent_id: RandomAgent(agent_id, env.action_space) for agent_id in env.possible_agents}
    
    # Reset environment
    observations, infos = env.reset()
    
    print(f"🎯 Mission: Find {len(env.survivor_positions)} survivors with weather effects!")
    
    for step in range(30):  # Run 30 steps
        actions = {agent_id: agent.act(observations[agent_id]) 
                   for agent_id, agent in agents.items() if agent_id in observations}
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render environment dynamically
        env.render()
        
        # Show total rewards
        total_reward = sum(rewards.values())
        print(f"💰 Step {step+1} Total Reward: {total_reward:+.2f}")
        
        if any(terminations.values()) or any(truncations.values()):
            print("\n✅ Mission Complete!")
            break
    
    print("\n🎉 Weather Demo completed!")
    print("=" * 40)


if __name__ == '__main__':
    run_simple_demo()  
    # run_weather_demo()  # Weather environment not yet implemented  
