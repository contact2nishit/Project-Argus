import pandas as pd
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

##read in hill data

def hill_conditions(hill_height, hill_width, hill_xstart , hill_ystart):
    
    hill = {
        "X_front" : hill_xstart, #where the hill begins on the x axis 
        "y_front" : hill_ystart, #where the hill begins on the y axis
        "X_end" : hill_xstart + hill_width,
        "y_end" : hill_ysrtast + hill_width,
        "height" : hill_height #how tall a hill is
    }

    return hill

    
class HillEnv(ParallelEnv):
    "multiple hill environment, including height"

    def __init__(self, num_agents = 3, grid_size = 10):
        self._num_agents = num_agents
        self.grid_size = grid_size
    

    self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
    self.agents = self.possible_agents[:]

    #observations and action spaces
    self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
    self.action_space = spaces.Discrete(5)

    #create 2 basic hills for now -> in the future read in data about the hills
    self.hills = [
        hills_conditions(5,5,5,5),
        hills_conditions(12,8,4,3)
    ]


    def reset(seed, seed=None, options=None):
        self.agents = self.possible_agents[:]

        #drone positions, z is the altitude of the drone
        self.positions = {}
        for agent in self.agents:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0,self.grid_size)
            z = np.random.randint(0,self.grid_size)

            self.positions[agent] = [x,y,z]

        def get_observation(self):
            obs = {}
            for agent in self.agents:
                x,y,z = self.positions[agent]

                #closest hill
                min_dist = float("inf")
                closest_hill = 'None'

                for hill in self.hills:

                    ##distance formula
                    hx = (hill["x_front"] + hill["x_end"]) / 2
                    hy = (hill["y_front"] + hill["y_end"]) / 2
                    dist = np.sqrt((hx - x) ** 2 + (hy - y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_hill = hill

            obs[agent] = np.array([x, y, z, nearest_hill["height"], min_dist], dtype=np.float32)
            return obs
        
        def step(self, actions):
            obs = {}
            rewards = {}
            terminations = {}
            truncations = {}
            infos = {}

            for agent in self.agents:
                x,y,z = self.positions[agent]
                if actions[agent] == 0:
                    z = -1
                elif actions[agent] ==1:
                    z +=1
                elif actions[agents] == 2:
                    x -=1
                elif actions[agent] == 3:
                    x +=1
                elif actions[agent] ==4:
                    y+=1
                

                #check collision with kills
                crashed = False
                for hill in self.hills:
                    ##if the position of the drone is in the area where the hill is
                    if hill["x_front"] <= x <= hill["x_end"] and hill["y_front"] <= y <= hill["y_end"]:
                        if z <= hill["height"]:
                            crashed = True
                            break
                
                if (crashed == True):
                    reward = 1
                else: 
                    reward = -.1
                
                done = crashed

                obs[agent] = np.array([x,y,z, hill["height"], 0], dtype = np.float32)
                rewards[agent] = reward
                terminations[agent] = done
                truncations[agent] = False
                infos[agent] = {}

        return obs, rewards, terminations, truncations, infos



                






















