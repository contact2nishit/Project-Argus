"""
Simple demo script for Project Argus

Basic demonstration of the rescue system.
"""

from src.random_agent import RandomAgent
from env.simple_rescue import SimpleRescueEnv


def run_simple_demo():
    """Run a basic demo with visualization."""
    print("🚁 Project Argus - Visual Demo 🚁\n")
    print("Watch the drones search for survivors!\n")
    time.sleep(2)
    
    # Create environment
    env = SimpleRescueEnv(num_agents=3, grid_size=8)
    
    # Create random agents
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = RandomAgent(agent_id, env.action_space)
    
    # Run one episode
    observations, infos = env.reset()
    
    # Show initial state
    env.render()
    print(f"🎯 Mission: Find {len(env.survivor_positions)} survivors!")
    print("⏸️  Starting in 2 seconds...\n")
    time.sleep(2)
    
    for step in range(20):  # Run 20 steps
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            if agent_id in observations:
                actions[agent_id] = agent.act(observations[agent_id])
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render the grid
        env.render()
        
        # Show rewards
        total_reward = sum(rewards.values())
        print(f"💰 Total Reward: {total_reward:+.2f}")
        
        # Show drone positions
        for agent_id in env.agents:
            pos = infos[agent_id]['position']
            print(f"   {agent_id}: ({pos[0]}, {pos[1]})")
        
        # Wait before next step
        time.sleep(1)  # 1 second delay between steps
        
        # Check if done
        if any(terminations.values()) or any(truncations.values()):
            print("\n✅ Mission Complete!")
            break
    
    print("\n🎉 Demo completed!")
    print("=" * 40)


if __name__ == '__main__':
    run_simple_demo()
