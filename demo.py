"""
Demo script for Project Argus

Includes a weather-based rescue simulation using snapshot CSV data.
"""

import os
import pandas as pd
from src.random_agent import RandomAgent
# from env.simple_rescue import SimpleRescueEnv
from env.simple_rescue import CSVRescueEnv  # Using CSVRescueEnv

def run_weather_demo_snapshot():
    """Run the weather-affected rescue demo with snapshot CSV."""
    print("🌤️ Project Argus - Weather Rescue Demo (CSV Snapshot) 🌤️\n")

    # Path to snapshot CSV
    snapshot_csv = os.path.join("env", "weather_data", "weather_snapshot_20251103_143543.csv")
    weather_df = pd.read_csv(snapshot_csv)

    # Create environment
    env = CSVRescueEnv(
        csv_file=snapshot_csv,       # Grid points
        weather_df=weather_df,       # Weather data
        num_agents=3
    )

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

        # Optionally render environment (you can add visualization later)
        # env.render()  

        # Show total rewards
        total_reward = sum(rewards.values())
        print(f"💰 Step {step+1} Total Reward: {total_reward:+.2f}")

        # Show drone positions
        for agent_id in env.agents:
            pos = infos[agent_id]['position']
            print(f"   {agent_id}: lat={pos[0]:.4f}, lon={pos[1]:.4f}")

        if any(terminations.values()) or any(truncations.values()):
            print("\n✅ Mission Complete!")
            break

    print("\n🎉 Weather Demo completed!")
    print("=" * 40)


if __name__ == '__main__':
    run_weather_demo_snapshot()
