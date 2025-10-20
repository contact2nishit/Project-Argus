from src.random_agent import RandomAgent
from env.test_env1 import TestEnv

def run_simple_demo():
    print("Project Argus - Simple Demo")
    env = TestEnv()
    agents = {agent_id: RandomAgent(agent_id, env.action_space) for agent_id in env.possible_agents}
    
    observations, infos = env.reset()
    
    for step in range(10):
        print(f"Step {step + 1}")
        actions = {agent_id: agent.act(observations[agent_id]) for agent_id, agent in agents.items() if agent_id in observations}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        total_reward = sum(rewards.values())
        print(f"  Total reward: {total_reward:.2f}")
        if any(terminations.values()):
            print("Episode terminated!")
            break

if __name__ == "__main__":
    run_simple_demo()