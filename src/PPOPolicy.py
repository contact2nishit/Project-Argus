import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
