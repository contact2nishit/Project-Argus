import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

class TestEnv(ParallelEnv):

    def __init__(self):
        ##petting zoo env with only two agents
   

        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        self.max_steps = 5
        self.current_step = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        obs = {agent: np.random.random(1).astype(np.float32) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        self.current_step += 1
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent in self.agents:
            observations[agent] = np.random.random(1).astype(np.float32)
            rewards[agent] = np.random.random()
            terminations[agent] = self.current_step >= self.max_steps
            truncations[agent] = False
            infos[agent] = {}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        print(f"Step {self.current_step}: Agents active = {self.agents}")

    def close(self):
        pass