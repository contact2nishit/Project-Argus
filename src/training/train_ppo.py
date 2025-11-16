"""
PPO Training Script for Simple Rescue Environment

Train a PPO agent to learn rescue operations on the grid-based environment.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environment.simple_rescue import SimpleRescueEnv
from src.ppo_agent import PPOAgent


def train_ppo(num_episodes=1000, max_steps=100, update_freq=1, save_freq=100):
    """Train PPO agent on simple rescue environment.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_freq: How often to update policy (in episodes) - should be 1 for PPO
        save_freq: How often to save the model
        
    Returns:
        agent: Trained PPO agent
        metrics: Training metrics
    """
    print("🚁 PPO Training on Simple Rescue Environment")
    print("=" * 60)
    
    # Create environment (single agent for now)
    env = SimpleRescueEnv(num_agents=1, grid_size=10)
    agent_id = env.possible_agents[0]
    
    print(f"📊 Environment Info:")
    print(f"   Grid size: {env.grid_size}x{env.grid_size}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n} actions")
    
    # Create PPO agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        agent_id=agent_id,
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=0.0003,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4
    )
    
    print(f"\n🤖 Agent Info:")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Device: {agent.device}")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    survivors_rescued = []
    
    print("\n🎓 Starting training...")
    print("=" * 60)
    
    episodes_since_update = 0
    
    for episode in range(num_episodes):
        # Reset environment
        observations, infos = env.reset()
        obs = observations[agent_id]
        
        episode_reward = 0
        rescued = 0
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.act(obs, training=True)
            
            # Step environment
            next_observations, rewards, terminations, truncations, next_infos = env.step({agent_id: action})
            
            reward = rewards[agent_id]
            done = terminations[agent_id] or truncations[agent_id]
            next_obs = next_observations[agent_id]
            
            # Store reward
            agent.store_reward(reward, done)
            
            # Track rescued survivors
            if next_infos[agent_id].get('rescued', False):
                rescued += 1
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        survivors_rescued.append(rescued)
        
        episodes_since_update += 1
        
        # Update policy periodically
        if episodes_since_update >= update_freq:
            agent.learn()
            episodes_since_update = 0
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_rescued = np.mean(survivors_rescued[-10:])
            
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Rescued: {avg_rescued:4.2f} | "
                  f"Steps: {avg_length:5.1f}")
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs('models', exist_ok=True)
            agent.save(f'models/ppo_agent_ep{episode + 1}.pt')
            print(f"   ✓ Model saved at episode {episode + 1}")
    
    print("=" * 60)
    print("✅ Training complete!")
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    agent.save('models/ppo_agent_final.pt')
    print("💾 Final model saved to models/ppo_agent_final.pt")
    
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'survivors_rescued': survivors_rescued
    }
    
    # Plot results
    plot_training_results(metrics)
    
    return agent, metrics


def plot_training_results(metrics, save_path='ppo_training_results.png'):
    """Plot training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(metrics['episode_rewards'], alpha=0.3, label='Episode')
    axes[0].plot(smooth(metrics['episode_rewards'], 20), linewidth=2, label='Avg (20 ep)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Survivors rescued
    axes[1].plot(metrics['survivors_rescued'], alpha=0.3, label='Episode')
    axes[1].plot(smooth(metrics['survivors_rescued'], 20), linewidth=2, label='Avg (20 ep)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Survivors Rescued')
    axes[1].set_title('Rescue Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Episode length
    axes[2].plot(metrics['episode_lengths'], alpha=0.3, label='Episode')
    axes[2].plot(smooth(metrics['episode_lengths'], 20), linewidth=2, label='Avg (20 ep)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')
    axes[2].set_title('Episode Length')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training plots saved to {save_path}")
    plt.close()


def smooth(data, window=10):
    """Smooth data using moving average."""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate trained agent.
    
    Args:
        agent: Trained PPO agent
        env: Environment to evaluate on
        num_episodes: Number of evaluation episodes
    """
    print("\n🎯 Evaluating Agent...")
    print("=" * 60)
    
    agent_id = env.possible_agents[0]
    
    total_rewards = []
    total_rescued = []
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        obs = observations[agent_id]
        
        episode_reward = 0
        rescued = 0
        
        for step in range(100):
            action = agent.act(obs, training=False)
            
            next_observations, rewards, terminations, truncations, next_infos = env.step({agent_id: action})
            
            reward = rewards[agent_id]
            done = terminations[agent_id] or truncations[agent_id]
            
            if next_infos[agent_id].get('rescued', False):
                rescued += 1
            
            episode_reward += reward
            obs = next_observations[agent_id]
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_rescued.append(rescued)
        
        print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Rescued={rescued}")
    
    print("=" * 60)
    print(f"📊 Evaluation Results:")
    print(f"   Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"   Avg Rescued: {np.mean(total_rescued):.2f} ± {np.std(total_rescued):.2f}")
    print("=" * 60)


if __name__ == '__main__':
    # Check PyTorch availability
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  Using CPU (training will be slower)\n")
    
    # Train agent
    agent, metrics = train_ppo(
        num_episodes=1000,
        max_steps=100,
        update_freq=1,  # Update after every episode for PPO
        save_freq=100
    )
    
    # Evaluate
    env = SimpleRescueEnv(num_agents=1, grid_size=10)
    evaluate_agent(agent, env, num_episodes=10)
