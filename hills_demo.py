from env.hillyEnv import HillyEnv
from src.single_agent import SimplePPOAgent
import numpy as np

def pad_or_clip_obs(obs, target_length=4):
    #debug chat so the input matches the observation length
    obs = np.array(obs, dtype=np.float32)
    if len(obs) > target_length:
        obs = obs[:target_length]  
    elif len(obs) < target_length:
        obs = np.concatenate([obs, np.zeros(target_length - len(obs))])  
    return obs

def run_hilly_demo():
    print("Project Argus Hilly Environment Demo")

    env = HillyEnv(num_agents=3, grid_size=10)
    obs_dict, _ = env.reset()
    agent_name = env.possible_agents[0]

    obs_size = 4
    action_size = env.action_space.shape[0]

    agent = SimplePPOAgent(obs_size, action_size)

    for step in range(10):
        actions = {}

        for a in env.agents:
            obs_fixed = pad_or_clip_obs(obs_dict[a], target_length=obs_size)
            if a == agent_name:
                actions[a] = agent.act(obs_fixed)
            else:
                actions[a] = np.random.uniform(
                    env.action_space.low, env.action_space.high, size=action_size
                )

        # Step the environment
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        # Print total reward
        total_reward = sum(rewards.values())
        print(f"\nStep {step+1}, Total reward: {total_reward:.2f}")

        # Print all agents, highlight PPO agent
        for a in env.agents:
            if a == agent_name:
                print(f"-> {a} (PPO): action = {actions[a]}")
            else:
                print(f"   {a} (Random): action = {actions[a]}")

        # Stop if any agent crashed
        if any(terminations.values()):
            print("crashed!")
            break

    print("\nDemo finished successfully!")

if __name__ == "__main__":
    run_hilly_demo()