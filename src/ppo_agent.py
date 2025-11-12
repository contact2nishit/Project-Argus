"""
PPO Agent for Simple Rescue Environment

Proximal Policy Optimization agent that learns rescue strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.base_agent import BaseAgent


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        """Initialize Actor-Critic network.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Number of discrete actions
            hidden_size: Size of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """Forward pass through network.
        
        Returns:
            action_probs: Probability distribution over actions
            state_value: Estimated value of the state
        """
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


class PPOAgent(BaseAgent):
    """PPO agent for learning optimal rescue strategies.
    
    PPO advantages:
    - Stable training with clipped objective
    - On-policy learning (learns from recent experience)
    - Works well with discrete action spaces
    - Natural exploration through stochastic policy
    """
    
    def __init__(self, agent_id, obs_dim, action_dim, 
                 learning_rate=0.0003, gamma=0.99, eps_clip=0.2,
                 K_epochs=4, value_coef=0.5, entropy_coef=0.01):
        """Initialize PPO agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_dim: Dimension of observation space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            eps_clip: Clipping parameter for PPO objective
            K_epochs: Number of epochs to update policy
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
        """
        super().__init__(agent_id)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic network
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Old policy for computing ratio
        self.policy_old = ActorCritic(obs_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # Loss function
        self.mse_loss = nn.MSELoss()
    
    def act(self, observation, training=True):
        """Select action using current policy.
        
        Args:
            observation: Current observation from environment
            training: Whether to store experience for training
            
        Returns:
            action: Selected action (integer)
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy_old(obs_tensor)
        
        # Sample action from distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store for training
        if training:
            self.states.append(observation)
            self.actions.append(action.item())
            self.logprobs.append(dist.log_prob(action).item())
            self.values.append(state_value.item())
        
        return action.item()
    
    def store_reward(self, reward, done):
        """Store reward and done flag.
        
        Args:
            reward: Reward received
            done: Whether episode is done
        """
        self.rewards.append(reward)
        self.dones.append(done)
    
    def learn(self, experience=None):
        """Update policy using PPO algorithm.
        
        This should be called at the end of each episode or after
        collecting a batch of experiences.
        """
        if len(self.states) == 0:
            return
        
        # Compute returns and advantages
        returns = []
        advantages = []
        discounted_reward = 0
        
        for reward, done, value in zip(reversed(self.rewards), 
                                       reversed(self.dones),
                                       reversed(self.values)):
            if done:
                discounted_reward = 0
            
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            advantages.insert(0, discounted_reward - value)
        
        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(self.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.logprobs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            
            # Compute ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Compute total loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(state_values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.clear_memory()
    
    def clear_memory(self):
        """Clear experience buffers."""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def save(self, filepath):
        """Save agent's policy network."""
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load agent's policy network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_old.load_state_dict(checkpoint['policy_old'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
