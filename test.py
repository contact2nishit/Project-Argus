from src.first_agent import Agent
from env.first_env import Env


def run():
    
    # Create environment
    env = Env(num_agents=5, grid_size=10)
    
    # Create random agents
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = Agent(agent_id, env.action_space)
    
    print(f"Created environment with {len(agents)} agents")
    
    # Run one episode
    observations, infos = env.reset()
    
    for step in range(50):
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
        print(f"  Total reward: {total_reward: .2f}")
        
        # Check if done
        if any(terminations.values()):
            print("Episode terminated!")
            break
    
    print("Demo completed!")


if __name__ == '__main__':
    run()
