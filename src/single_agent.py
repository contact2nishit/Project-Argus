import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimplePPOAgent:
    def __init__(self, obs_size, action_size, lr=1e-3):
        #brain to decide on actions
        self.policy = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh() 
        )

        #brain for rewards -> what is good/bad
        self.value = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr) #what the agent did vs what it should have done, adjustts weights
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr) #same but for estimating rewards

    def act(self, obs):
        #takes what agent sees and returns what it should do 
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :] 

        obs_tensor = torch.from_numpy(obs)
        with torch.no_grad():
            action = self.policy(obs_tensor)
        return action.squeeze(0).numpy()  

    def get_value(self, obs):
        #how good is the current action/state -> so it says whether to keep or change and action during learning
        #converts a single number tensor to a regular python number
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_tensor = torch.from_numpy(obs)
        with torch.no_grad():
            value = self.value(obs_tensor)
        return value.squeeze(0).item()

    def learn(self, trajectories):
        pass