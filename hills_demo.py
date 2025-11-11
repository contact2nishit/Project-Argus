from src.first_agent import single_agent
from env.hillyEnv import HillyEnv

def run_hilly_demo():
    print("Project Argus hilly environment demo")

    env = HillyEnv(num_agents = 3, grid_size = 10)

    agents = {
        agent_id: RandomAgent(agent_id, env.action_space)
        for agent_id in env.possible_agents
    }

    print(f"Created an environment with {len(agents)} drones and {len(env.hills)} hills")

    observations, infos = env.reset()

    for step in range(10):
        print(f"\nStep {step + 1}")

        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.act(observations[agent_id])
        
        observations, rewards, terminations, truncations, infos = env.step(actions)

        total_reward = sum(rewards.values())
        print(f"Total reward this step: {total_reward:.2f}")

        if any(terminations.values()):
            print("drone has crashed")
            break
    
    print("\nDemo finished sucessfully!")

if __name__ == '__main__':
    run_hilly_demo()

