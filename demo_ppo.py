"""
PPO Training Demo for Project Argus

Demonstrates PPO agents learning in the rescue environment with live results graphing.
"""

import matplotlib.pyplot as plt
import numpy as np
from src.base_agent_PPO import PPAgent
from env.simple_rescue import SimpleRescueEnv


class PPOAgent(PPAgent):
    """Concrete PPO agent implementation."""
    
    def __init__(self, state_size, action_size, agent_id, **kwargs):
        super().__init__(state_size, action_size, agent_id, **kwargs)
        self.last_log_prob = None
    
    def act(self, observation):
        """Select an action given an observation using the PPO policy."""
        action, log_prob = self.select_action(observation)
        self.last_log_prob = log_prob
        return action
    
    def learn(self, experience):
        """Store experience and train when batch is full."""
        state, action, reward, next_state, done = experience
        
        # Store the experience with the last log probability
        self.store_experience(state, action, reward, next_state, done, self.last_log_prob)
        
        # Train when we have enough experiences
        if len(self.memory) >= self.batch_size:
            self.train()


def run_ppo_training():
    """Run PPO training with live results plotting."""
    print("🚁 Project Argus - PPO Training Demo! 🚁")
    
    # Environment setup
    num_agents = 3
    env = SimpleRescueEnv(num_agents=num_agents, grid_size=50, render_mode=None, num_survivors=2)
    
    # Get state and action sizes
    observations, infos = env.reset(use_noise=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create PPO agents
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id=agent_id,
            learning_rate=3e-4,
            gamma=0.99,
            clip_param=0.2,
            batch_size=64,
            ppo_epochs=4
        )
    
    print(f"Created {len(agents)} PPO agents")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Training parameters
    num_episodes = 50
    max_steps_per_episode = 100
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    avg_rewards_window = []
    
    # Setup live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Run training episodes
    for episode in range(num_episodes):
        observations, infos = env.reset(use_noise=True)
        episode_reward = 0
        step = 0
        
        for step in range(max_steps_per_episode):
            # Get actions from all agents
            actions = {}
            for agent_id, agent in agents.items():
                if agent_id in observations:
                    actions[agent_id] = agent.act(observations[agent_id])
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store experiences for learning
            for agent_id, agent in agents.items():
                if agent_id in observations:
                    done = terminations[agent_id] or truncations[agent_id]
                    agent.learn((
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_observations[agent_id],
                        done
                    ))
            
            # Track rewards
            episode_reward += sum(rewards.values())
            observations = next_observations
            
            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Calculate moving average
        window_size = min(10, len(episode_rewards))
        avg_reward = np.mean(episode_rewards[-window_size:])
        avg_rewards_window.append(avg_reward)
        
        # Print progress
        memory_sizes = [len(agent.memory) for agent in agents.values()]
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Steps: {step + 1} | "
              f"Memory: {memory_sizes}")
        
        # Update plots every 5 episodes
        if (episode + 1) % 5 == 0 or episode == 0:
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Plot 1: Episode rewards with moving average
            ax1.plot(episode_rewards, label='Episode Reward', alpha=0.6, color='blue')
            ax1.plot(avg_rewards_window, label='Moving Avg (10 episodes)', 
                    linewidth=2, color='red')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('PPO Training Progress - Rewards Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Episode lengths
            ax2.plot(episode_lengths, label='Episode Length', color='green')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.set_title('Episode Lengths Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
    
    # Final results
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Total Episodes: {num_episodes}")
    print(f"Final Average Reward (last 10): {avg_rewards_window[-1]:.2f}")
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    print("="*60)
    
    # Keep plot open
    plt.ioff()
    plt.show()
    
    env.close()


def run_visual_demo():
    """Run a single episode with visualization using trained agents."""
    print("\n🚁 Running Visual Demo with PPO Agents! 🚁")
    
    # Environment setup with rendering
    num_agents = 3
    env = SimpleRescueEnv(num_agents=num_agents, grid_size=50, render_mode="human", num_survivors=2)
    
    # Get state and action sizes
    observations, infos = env.reset(use_noise=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create PPO agents (these will be untrained for demo)
    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id=agent_id,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64
        )
    
    print(f"Running visual demo with {len(agents)} agents...")
    
    # Run one episode with rendering
    env.render()
    
    for step in range(50):
        env.render()
        
        # Get actions
        actions = {}
        for agent_id, agent in agents.items():
            if agent_id in observations:
                actions[agent_id] = agent.act(observations[agent_id])
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"Step {step + 1} | Total Reward: {sum(rewards.values()):.2f}")
        
        if any(terminations.values()) or any(truncations.values()):
            print("Episode complete!")
            break
    
    env.close()
    print("Visual demo completed!")


if __name__ == '__main__':
    # Choose which demo to run
    print("Select demo mode:")
    print("1. PPO Training with live graphs (no visualization)")
    print("2. Visual demo with rendering (single episode)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    if choice == '2':
        run_visual_demo()
    else:
        run_ppo_training()
