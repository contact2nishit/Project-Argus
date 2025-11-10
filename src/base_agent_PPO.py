"""
Base Agent Interface for Project Argus

Simple abstract base class for all RL agents.
"""

from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    """Actor network"""
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # Defining network layers: Input --> Hidden --> ReLU --> --> Output
        self.input_layer = nn.Linear(state_size, 128) # Input layer
        self.layer1 = nn.Linear(128, 128) 
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_size)

    def forward(self, state):
        """Forward pass through the network."""

        # Applies ReLU activation after each hidden layer because it introduces non-linearity
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        # Converts network output to action probabilities with softmax function
        action_probs = torch.softmax(self.output_layer(x), dim=-1)
        return action_probs
    
class Critic(nn.Module):
    """Critic network: determines value of state --> 'How good a state is' """

    def __init__(self, state_size):
        super(Critic, self).__init__()
        # Defining network layers: Input --> Hidden --> ReLU --> --> Output
        self.input_layer = nn.Linear(state_size, 128) # Input layer
        self.layer1 = nn.Linear(128, 128) 
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1)  # Outputs a single value

    def forward(self, state):
        """Forward pass through the network."""

        # Applies ReLU activation after each hidden layer because it introduces non-linearity
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        # Outputs a single scalar value representing the state's value 
        state_value = self.output_layer(x) # No activation or softmax here
        return state_value

class PPAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self,  state_size, action_size, agent_id: str, learning_rate=1e-3, gamma=0.99, clip_param=0.2, batch_size=2048, ppo_epochs=10):
        # Initialize networks
        self.agent_id = agent_id
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)

        # Optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Define hyperparameters
        self.gamma = gamma  # Discount factor, how much future rewards are valued
        self.clip_param = clip_param  # PPO clipping parameter, prevents large policy updates
        self.batch_size = batch_size  # Number of experiences to use for each training update
        self.ppo_epochs = ppo_epochs  # Number of epochs to train over each batch

        # Create memory, to store information for one batch
        self.memory = []
    
    def select_action(self, state):
        """Selects action based on policy and a state."""
        # Convert state to tensor for usability
        state = torch.FloatTensor(state).unsqueeze(0)  

        # Get action probabilities from actor network
        with torch.no_grad():
            action_probs = self.actor(state)

        # Create a categorical distribution over the action probabilities and sample
        dist = Categorical(action_probs)
        action = dist.sample()

        # Get log probability of the selected action
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def store_experience(self, state, action, reward, next_state, done, log_prob):
        # Store the experience tuple
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def train(self):
        """Train the agent using experiences in memory."""
        
        # Extract states, actions, rewards, dones, and old log probabilities from memory

        states = [mem[0] for mem in self.memory]
        actions = [mem[1] for mem in self.memory]
        rewards = [mem[2] for mem in self.memory]
        dones = [mem[4] for mem in self.memory]
        old_log_probs = [mem[5] for mem in self.memory]
        
        # Calculate Reward-to-go
        discounted_reward = 0
        returns = []

        for timestep in range(-1, -len(self.memory), -1):
            if self.memory[timestep][4]: # Checks if episode done
                discounted_reward = 0
            
            discounted_reward = self.memory[_][2] + (self.gamma * discounted_reward)
            returns.append(discounted_reward)
            
        returns.reverse()

        # Convert extracted data to tensors, blanket squeezes to remove unnecessary extra dimensions
        states_tensor = torch.FloatTensor(states).squeeze()
        actions_tensor = torch.LongTensor(actions).squeeze()
        returns_tensor = torch.FloatTensor(returns).squeeze()
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).squeeze()

        # Call critic on all states
        with torch.no_grad():
            values_tensor = self.critic(states_tensor).squeeze()

        # Formulate advantages
        advantages_tensor = returns_tensor - values_tensor

        # Normalize advantages for stability (Optional)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO training epochs - How many times the data is analyzed and fed through the networks
        for _ in range(self.ppo_epochs):
            # Re-evaluate states by passing through the critic and actor networks
            states_tensor = self.critic(states_tensor).squeeze()
            new_probs = self.actor(states_tensor)
            new_dist = Categorical(new_probs)
            new_log_probs = new_dist.log_prob(actions_tensor)

            # Calculate Actor loss

            # Compute ratio of new and old policy probabilities: >1 means new policy is favored, and <1 means old policy is favored
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # Compute surrogate losses
            surr1 = ratio * advantages_tensor # "Greedy" policy update, encourages large changes
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_tensor # "Safe" policy update, prevents large changes and ensure stability

            # Calculate actor loss using the minimum of the two surrogate losses
            actor_loss = -torch.min(surr1, surr2).mean() # Actor loss is negative because we want to minimize the loss

            # Calculate Critic loss
            critic_loss = ((returns_tensor - values_tensor) ** 2).mean()  # Mean Squared Error loss for critic (How far off the value estimate was from the actual reward)

            # Update Actor & Critic networks --> The learning function, feeds back the losses to the optimizers
            self.actor_optimizer.zero_grad() # Clear previous optimizer gradients
            actor_loss.backward()  # Backpropagate actor loss (feed new actor info through network)
            self.actor_optimizer.step()  # Update actor network parameters (actor optimizer gets a tad smarter)

            self.critic_optimizer.zero_grad()  # Clear previous optimizer gradients
            critic_loss.backward()  # Backpropagate critic loss (feed new critic info through network)
            self.critic_optimizer.step()  # Update critic network parameters (critic optimizer gets a tad smarter)

            # Epoch Complete
            

        # Clear Memory after training
        self.memory = []

        pass
    
    @abstractmethod
    def act(self, observation):
        """Select an action given an observation."""
        pass
    
    @abstractmethod
    def learn(self, experience):
        """Update the agent's policy."""
        pass