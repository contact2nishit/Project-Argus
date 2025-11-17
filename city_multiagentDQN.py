import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from env.city_env import BuildingEnv
from src.DQN_agent import DQNAgent


def smooth(data, window=10):
    """Smooth data using moving average."""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


def train_dqn(num_episodes=1000, max_steps=100, save_freq=100):
    """Train DQN agents on the building environment."""
    print("DQN Training on Building Environment")
    print("=" * 60)

    env = BuildingEnv(num_agents=3, grid_size=10)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize one DQN agent per drone
    agents = {}
    for agent_name in env.possible_agents:
        agents[agent_name] = DQNAgent(agent_name, obs_dim, action_dim, lr=0.003)

    # Metrics
    episode_rewards = []
    episode_lengths = []
    survivors_rescued = []

    print(" Starting training...")
    print("=" * 60)

    for episode in range(num_episodes):
        obs_dict, infos = env.reset()
        episode_reward = 0
        rescued_count = 0  # Counter for rescued survivors

        for step in range(max_steps):
            actions = {}
            for agent_name, agent in agents.items():
                obs = np.array(obs_dict[agent_name], dtype=np.float32)
                actions[agent_name] = agent.act(obs, training=True)

            next_obs_dict, rewards, terminations, truncations, next_infos = env.step(actions)

            for agent_name, agent in agents.items():
                agent.store_transition(
                    obs_dict[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    next_obs_dict[agent_name],
                    terminations[agent_name]
                )

                # Update per agent every few steps
                if step % 4 == 0:
                    agent.learn()

            # Increment rescued count properly
            for agent_info in next_infos.values():
                if agent_info.get("rescued", False):
                    rescued_count += 1

            episode_reward += sum(rewards.values())
            obs_dict = next_obs_dict

            if any(terminations.values()):
                break

        # Update target networks after each episode
        for agent in agents.values():
            agent.update_target()

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        survivors_rescued.append(rescued_count)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_rescued = np.mean(survivors_rescued[-10:])
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Rescued: {avg_rescued:4.2f} | "
                  f"Steps: {avg_length:5.1f}")

        # Save models periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs('models', exist_ok=True)
            for agent_name, agent in agents.items():
                agent.save(f'models/dqn_{agent_name}_ep{episode + 1}.pt')
            print(f"   ✓ Models saved at episode {episode + 1}")

    print("=" * 60)
    print(" Training complete!")

    # Save final models
    os.makedirs('models', exist_ok=True)
    for agent_name, agent in agents.items():
        agent.save(f'models/dqn_{agent_name}_final.pt')
    print("Final models saved to models/")

    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'survivors_rescued': survivors_rescued
    }

    plot_training_results(metrics)
    return agents, metrics


def plot_training_results(metrics, save_path='dqn_training_results.png'):
    """Plot DQN training metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    episodes = range(len(metrics['episode_rewards']))

    # Rewards
    axes[0].plot(episodes, metrics['episode_rewards'], alpha=0.3, label='Episode')
    axes[0].plot(episodes, smooth(metrics['episode_rewards'], 20), linewidth=2, label='Avg (20 ep)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Survivors rescued
    axes[1].plot(episodes, metrics['survivors_rescued'], alpha=0.3, label='Episode')
    axes[1].plot(episodes, smooth(metrics['survivors_rescued'], 20), linewidth=2, label='Avg (20 ep)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Survivors Rescued')
    axes[1].set_title('Rescue Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Episode lengths
    axes[2].plot(episodes, metrics['episode_lengths'], alpha=0.3, label='Episode')
    axes[2].plot(episodes, smooth(metrics['episode_lengths'], 20), linewidth=2, label='Avg (20 ep)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')
    axes[2].set_title('Episode Length')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f" Training plots saved to {save_path}")
    plt.close()


def evaluate_agents(agents, num_episodes=10, max_steps=100):
    """Evaluate trained DQN agents."""
    print("\n Evaluating Agents...")
    print("=" * 60)

    env = BuildingEnv(num_agents=3, grid_size=10)

    total_rewards = []
    total_rescued = []

    for episode in range(num_episodes):
        obs_dict, infos = env.reset()
        episode_reward = 0
        rescued_count = 0

        for step in range(max_steps):
            actions = {}
            for agent_name, agent in agents.items():
                obs = np.array(obs_dict[agent_name], dtype=np.float32)
                actions[agent_name] = agent.act(obs, training=False)

            next_obs_dict, rewards, terminations, truncations, next_infos = env.step(actions)

            # Count rescued properly
            for agent_info in next_infos.values():
                if agent_info.get("rescued", False):
                    rescued_count += 1

            episode_reward += sum(rewards.values())
            obs_dict = next_obs_dict

            if any(terminations.values()):
                break

        total_rewards.append(episode_reward)
        total_rescued.append(rescued_count)
        print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Rescued={rescued_count}")

    print("=" * 60)
    print(f" Evaluation Results:")
    print(f"   Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"   Avg Rescued: {np.mean(total_rescued):.2f} ± {np.std(total_rescued):.2f}")
    print("=" * 60)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("  Using CPU\n")

    agents, metrics = train_dqn(
        num_episodes=1000,
        max_steps=100,
        save_freq=100
    )

    evaluate_agents(agents, num_episodes=10, max_steps=100)
