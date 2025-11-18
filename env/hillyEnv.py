import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

def hill_heights(x, y, hills):
    z = 0
    for hill in hills:
        dx = x - hill["x_c"]
        dy = y - hill["y_c"]
        z += hill["h"] * np.exp(-(dx**2 + dy**2) / (2 * hill["sigma"]**2))
    return z

class HillyEnv(ParallelEnv):
    def __init__(self, num_agents = 3, grid_size = 10):
        self._num_agents = num_agents
        self.grid_size = grid_size
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        #observations
        self.observations_space = spaces.Box(low = 0, high = 100, shape = (5,), dtype = np.float32)
        #actions:beug don't make it discrete anymore -> changein x,y,and z dx,dy,dz
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (3,), dtype = np.float32)

        self.hills = [
            {"x_c" : 5, "y_c": 5, "h":5, "sigma": 2},
            {"x_c": 2, "y_c": 7, "h":3, "sigma":1.5}
        ]


    def reset(self, seed = None, options = None):
        self.agents = self.possible_agents[:]

        #we can start at random pos above the terrain
        self.positions = {
            agent: np.array([
                np.random.uniform(0, self.grid_size),
                np.random.uniform(0, self.grid_size),
                np.random.uniform(5,10)
            ])for agent in self.agents
        }

        num_survivors = np.random.randint(2,4)
        self.survivors = [np.array([np.random.uniform(0,self.grid_size),
        np.random.uniform(0,self.grid_size)]) for _ in range(num_survivors)]

        obs = self.get_observations()
        infos = {agent: {} for agent in self.agents}
        return obs,infos

    def get_observations(self):
        obs = {}
        for agent in self.agents:
            x, y, z = self.positions[agent]
            hill_z = hill_heights(x, y, self.hills)
            
            if len(self.survivors) > 0:
                nearest_dist = min(np.linalg.norm(self.positions[agent][:2] - survivor) for survivor in self.survivors)
            else:
                nearest_dist = 0.0
                
            obs[agent] = np.array([x, y, z, hill_z, nearest_dist], dtype=np.float32) 
        return obs
    
    def step(self, actions):
        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}


        for agent in self.agents:
            dx, dy, dz = actions[agent]
            self.positions[agent] += np.array([dx, dy, dz])

            x,y,z = self.positions[agent]
            hill_z =hill_heights(x, y, self.hills)

            #collision checker
            crashed = (z<=hill_z)

            if (crashed == True):
                reward = -1.0
            else:
                reward = 1.0

            ##check if it is able to be near a survivor
            survivor_found = False
            nearest_distance = 0.0

            if len(self.survivors) > 0:

                i = 0
                while i < len(self.survivors):
                    survivor = self.survivors[i]
                    dist = np.linalg.norm(self.positions[agent][:2] - survivor)
                    nearest_distance = min(nearest_distance, dist)
                    ##debug: if it is near the survivor
                    if dist < 1.0: 
                        survivor_found = True
                        self.survivors.pop(i)
                    else:
                        i += 1
            
            if survivor_found == True:
                reward += 5.0

            else:
                if len(self.survivors) > 0:
                    #debug: smaller distance bigger the award
                    reward += 0.1 * (1.0/ (nearest_distance + 1.0))


            
            rewards[agent] = reward
            terminations[agent] = crashed
            truncations[agent] = False
            infos[agent] = {}
        
        


        obs = self.get_observations()
        return obs, rewards, terminations, truncations, infos


        



