import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from env.hillyEnv import HillyEnv
from src.sac_agent import SACAgent

def smooth(data, window=10):
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed

def train_sac():
    print("SAC Training on Hilly Environment")
    print("=" * 50)
    
    env = HillyEnv(num_agents=3, grid_size=10)
    obs_size = 5
    action_size = env.action_space.shape[0]
    
    agents = {}
    for agent_name in env.possible_agents:
        agents[agent_name] = SACAgent(agent_name, obs_size, action_size)
    
    episode_rewards = []
    survivors_rescued = []
    collision_rates = []
    
    for episode in range(1000):
        obs_dict, _ = env.reset()
        initial_survivors = len(env.survivors)
        total_reward = 0
        collisions = 0
        
        for step in range(100):
            actions = {}
            for agent_name, agent in agents.items():
                obs = np.array(obs_dict[agent_name], dtype=np.float32)
                if len(obs) > obs_size:
                    obs = obs[:obs_size]
                actions[agent_name] = agent.act(obs)
            
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent_name, agent in agents.items():
                agent.store_transition(
                    obs_dict[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    next_obs_dict[agent_name],
                    terminations[agent_name]
                )
            
            total_reward += sum(rewards.values())
            collisions += sum(terminations.values())
            obs_dict = next_obs_dict
            
            if any(terminations.values()):
                break
        
        for agent in agents.values():
            agent.learn()
        
        rescued = initial_survivors - len(env.survivors)
        collision_rate = (collisions / len(env.agents)) * 100
        
        episode_rewards.append(total_reward)
        survivors_rescued.append(rescued)
        collision_rates.append(collision_rate)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_rescued = np.mean(survivors_rescued[-50:])
            avg_collision = np.mean(collision_rates[-50:])
            
            print(f"Episode {episode + 1:4d} | Reward: {avg_reward:7.2f} | "
                  f"Rescued: {avg_rescued:4.2f} | Collision: {avg_collision:5.1f}%")
        
        if (episode + 1) % 200 == 0:
            os.makedirs('models', exist_ok=True)
            for agent_name, agent in agents.items():
                agent.save(f'models/sac_{agent_name}_ep{episode + 1}.pt')
            print(f"Models saved at episode {episode + 1}")
    
    os.makedirs('models', exist_ok=True)
    for agent_name, agent in agents.items():
        agent.save(f'models/sac_{agent_name}_final.pt')
    
    metrics = {
        'episode_rewards': episode_rewards,
        'survivors_rescued': survivors_rescued,
        'collision_rates': collision_rates
    }
    
    plot_results(metrics)
    return agents, metrics

def plot_results(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    episodes = range(len(metrics['episode_rewards']))
    
    axes[0].plot(episodes, metrics['episode_rewards'], alpha=0.3, label='Episode')
    axes[0].plot(episodes, smooth(metrics['episode_rewards'], 20), linewidth=2, label='Avg (20 ep)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(episodes, metrics['survivors_rescued'], alpha=0.3, label='Episode')
    axes[1].plot(episodes, smooth(metrics['survivors_rescued'], 20), linewidth=2, label='Avg (20 ep)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Survivors Rescued')
    axes[1].set_title('Rescue Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(episodes, metrics['collision_rates'], alpha=0.3, label='Episode')
    axes[2].plot(episodes, smooth(metrics['collision_rates'], 20), linewidth=2, label='Avg (20 ep)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Collision Rate (%)')
    axes[2].set_title('Collision Rate')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sac_training_results.png', dpi=150)
    plt.close()

def evaluate_agents(agents, num_episodes=10):
    print("Evaluating SAC Agents")
    print("=" * 50)
    
    env = HillyEnv(num_agents=3, grid_size=10)
    obs_size = 5
    
    test_rewards = []
    test_rescued = []
    
    for episode in range(num_episodes):
        obs_dict, _ = env.reset()
        initial_survivors = len(env.survivors)
        total_reward = 0
        
        for step in range(100):
            actions = {}
            for agent_name, agent in agents.items():
                obs = np.array(obs_dict[agent_name], dtype=np.float32)
                if len(obs) > obs_size:
                    obs = obs[:obs_size]
                actions[agent_name] = agent.act(obs, deterministic=True)
            
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            obs_dict = next_obs_dict
            
            if any(terminations.values()):
                break
        
        rescued = initial_survivors - len(env.survivors)
        test_rewards.append(total_reward)
        test_rescued.append(rescued)
        print(f"Test {episode + 1}: Reward={total_reward:.2f}, Rescued={rescued}")
    
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Average Rescued: {np.mean(test_rescued):.2f}")

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    trained_agents, metrics = train_sac()
    evaluate_agents(trained_agents)