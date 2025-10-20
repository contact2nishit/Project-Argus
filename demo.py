"""
Simple demo script for Project Argus

Basic demonstration of the rescue system.
"""
from pettingzoo.mpe import simple_spread_v3 
from pettingzoo.utils import aec_to_parallel
from src.random_agent import RandomAgent
from env.simple_rescue import SimpleRescueEnv


def run_simple_demo():
    """Run a basic demo."""
    print("🚁 Project Argus - Simple Demo! 🚁")
    
    # Create pettingzoo environment
    aec_env = simple_spread_v3.env(N=3, max_cycles=25, render_mode = "human")

    # Converts aec environment to parallel environment
    env = aec_to_parallel(aec_env)

    
    # Initialize the environment
    env.reset() 
    
    # Create random agents
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = RandomAgent(agent_id, env.action_space(agent_id))
    
    print(f"Created environment with {len(agents)} agents")
    
    # Run one episode
    observations, infos = env.reset()
    
    for step in range(10):  # Run 10 steps
        print(f"Step {step + 1}")
        
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            if agent_id in observations:
                actions[agent_id] = agent.act(observations[agent_id])
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Print step info
        total_reward = sum(rewards.values())
        print(f"  Total reward: {total_reward:.2f}")
        
        # Check if done
        if any(terminations.values()):
            print("Episode terminated!")
            break
    
    print("Demo completed!")


if __name__ == '__main__':
    run_simple_demo()
