"""
Demo script for trained PPO agent

Visualize the trained PPO agent performing rescue operations.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environment.simple_rescue import SimpleRescueEnv
from src.ppo_agent import PPOAgent


def visualize_episode(env, agent, delay=0.5):
    """Run and visualize one episode.
    
    Args:
        env: Environment
        agent: Trained PPO agent
        delay: Delay between steps (seconds)
    """
    agent_id = env.possible_agents[0]
    observations, infos = env.reset()
    obs = observations[agent_id]
    
    print("\n" + "=" * 60)
    print("🎬 Starting Episode")
    print("=" * 60)
    
    # Print initial state
    print(f"\n📍 Initial State:")
    print(f"   Agent position: {infos[agent_id]['position']}")
    print(f"   Survivors: {len(env.survivor_positions)}")
    for i, pos in enumerate(env.survivor_positions):
        print(f"      Survivor {i+1}: {pos}")
    
    episode_reward = 0
    rescued_count = 0
    
    action_names = ["UP ⬆️ ", "DOWN ⬇️ ", "LEFT ⬅️ ", "RIGHT ➡️ "]
    
    print(f"\n🎮 Actions:")
    
    for step in range(100):
        # Get action
        action = agent.act(obs, training=False)
        
        # Step environment
        next_observations, rewards, terminations, truncations, next_infos = env.step({agent_id: action})
        
        reward = rewards[agent_id]
        done = terminations[agent_id] or truncations[agent_id]
        next_obs = next_observations[agent_id]
        next_pos = next_infos[agent_id]['position']
        dist = next_infos[agent_id]['nearest_survivor_distance']
        
        # Check if rescued
        rescued = next_infos[agent_id].get('rescued', False)
        if rescued:
            rescued_count += 1
        
        # Print step info
        status = ""
        if rescued:
            status = "🎉 RESCUED SURVIVOR!"
        elif reward > 0.05:
            status = f"✓ Moving closer (dist: {dist})"
        elif reward < -0.05:
            status = f"✗ Moving away (dist: {dist})"
        else:
            status = f"→ No change (dist: {dist})"
        
        print(f"   Step {step+1:3d}: {action_names[action]} → {next_pos} | "
              f"Reward: {reward:+6.2f} | {status}")
        
        episode_reward += reward
        obs = next_obs
        
        if delay > 0:
            time.sleep(delay)
        
        if done:
            print(f"\n✅ All survivors rescued in {step+1} steps!")
            break
    
    print("\n" + "=" * 60)
    print(f"📊 Episode Summary:")
    print(f"   Total Reward: {episode_reward:.2f}")
    print(f"   Steps Taken: {step+1}")
    print(f"   Survivors Rescued: {rescued_count}/{len(env.survivor_positions)}")
    print(f"   Efficiency: {episode_reward/(step+1):.3f} reward/step")
    print("=" * 60)


def compare_random_vs_ppo(num_episodes=10):
    """Compare random policy vs trained PPO.
    
    Args:
        num_episodes: Number of episodes to compare
    """
    print("\n🆚 Comparison: Random Policy vs PPO Agent")
    print("=" * 60)
    
    env = SimpleRescueEnv(num_agents=1, grid_size=10)
    agent_id = env.possible_agents[0]
    
    # Load trained agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ppo_agent = PPOAgent(agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim)
    
    try:
        ppo_agent.load('models/ppo_agent_final.pt')
        print("✓ Loaded trained PPO model\n")
    except FileNotFoundError:
        print("⚠️  No trained model found. Train first with: python3 train_ppo.py\n")
        return
    
    # Test random policy
    print("🎲 Testing Random Policy...")
    random_rewards = []
    random_rescued = []
    
    for ep in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0
        rescued = 0
        
        for step in range(100):
            action = np.random.randint(0, 4)  # Random action
            observations, rewards, terminations, truncations, infos = env.step({agent_id: action})
            
            reward = rewards[agent_id]
            done = terminations[agent_id]
            
            if infos[agent_id].get('rescued', False):
                rescued += 1
            
            episode_reward += reward
            
            if done:
                break
        
        random_rewards.append(episode_reward)
        random_rescued.append(rescued)
    
    # Test PPO agent
    print("🤖 Testing PPO Agent...")
    ppo_rewards = []
    ppo_rescued = []
    
    for ep in range(num_episodes):
        observations, infos = env.reset()
        obs = observations[agent_id]
        episode_reward = 0
        rescued = 0
        
        for step in range(100):
            action = ppo_agent.act(obs, training=False)
            observations, rewards, terminations, truncations, infos = env.step({agent_id: action})
            
            reward = rewards[agent_id]
            done = terminations[agent_id]
            obs = observations[agent_id]
            
            if infos[agent_id].get('rescued', False):
                rescued += 1
            
            episode_reward += reward
            
            if done:
                break
        
        ppo_rewards.append(episode_reward)
        ppo_rescued.append(rescued)
    
    # Print comparison
    print("\n📊 Comparison Results ({} episodes):".format(num_episodes))
    print("=" * 60)
    print(f"{'Metric':<25} {'Random':<15} {'PPO':<15} {'Improvement'}")
    print("-" * 60)
    
    avg_random_reward = np.mean(random_rewards)
    avg_ppo_reward = np.mean(ppo_rewards)
    reward_improvement = ((avg_ppo_reward - avg_random_reward) / abs(avg_random_reward) * 100) if avg_random_reward != 0 else 0
    print(f"{'Avg Reward':<25} {avg_random_reward:<15.2f} {avg_ppo_reward:<15.2f} {reward_improvement:+.1f}%")
    
    avg_random_rescued = np.mean(random_rescued)
    avg_ppo_rescued = np.mean(ppo_rescued)
    rescued_improvement = ((avg_ppo_rescued - avg_random_rescued) / avg_random_rescued * 100) if avg_random_rescued != 0 else 0
    print(f"{'Avg Survivors Rescued':<25} {avg_random_rescued:<15.2f} {avg_ppo_rescued:<15.2f} {rescued_improvement:+.1f}%")
    
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    # Create environment
    env = SimpleRescueEnv(num_agents=1, grid_size=10)
    agent_id = env.possible_agents[0]
    
    # Load agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim)
    
    try:
        agent.load('models/ppo_agent_final.pt')
        print("✓ Loaded trained PPO model")
    except FileNotFoundError:
        print("⚠️  No trained model found. Train first with:")
        print("   python3 train_ppo.py")
        sys.exit(1)
    
    # Run visualization
    print("\n🎬 Running Demo Episodes...\n")
    
    for i in range(3):
        print(f"\n{'='*60}")
        print(f"EPISODE {i+1}/3")
        visualize_episode(env, agent, delay=0.3)
    
    # Run comparison
    print("\n")
    compare_random_vs_ppo(num_episodes=20)
