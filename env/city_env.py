import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

##anytime you read in building here -> think of it as more like a building/rectangular figure

def building_conditions(building_height, building_width, building_xstart , building_ystart):
    
    building = {
        "X_front" : building_xstart, #where the building begins on the x axis 
        "y_front" : building_ystart, #where the buildingbegins on the y axis
        "X_end" : building_xstart + building_width,
        "y_end" : building_ystart + building_width,
        "height" : building_height #how tall a building is
    }

    return building

    
class BuildingEnv(ParallelEnv):
    "multiple building environment, including height"

    def __init__(self, num_agents = 3, grid_size = 10):
        self._num_agents = num_agents
        self.grid_size = grid_size
    

        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        #observations and action spaces
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        #create 2 basic buildings for now -> in the future read in data about the buildings
        self.buildings = [
            building_conditions(5,5,5,5),
            building_conditions(12,8,4,3)
        ]


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        #drone positions, z is the altitude of the drone
        self.positions = {}
        for agent in self.agents:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0,self.grid_size)
            z = np.random.randint(0,self.grid_size)

            self.positions[agent] = [x,y,z]

        #add a position for one random survivor
        self.survivor = [np.random.randint(0,self.grid_size), np.random.randint(0, self.grid_size), np.random.randint(0,self.grid_size)]


        
        obs = {}
        for agent in self.agents:
            x,y,z = self.positions[agent]

            #closest building
            min_dist = float("inf")
            nearest_building = 'None'

            for building in self.buildings:

                ##distance formula
                hx = (building["X_front"] + building["X_end"]) / 2
                hy = (building["y_front"] + building["y_end"]) / 2
                dist = np.sqrt((hx - x) ** 2 + (hy - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_building = building

            obs[agent] = np.array([x, y, z, nearest_building["height"], min_dist], dtype=np.float32)
        
        infos = {agent: {} for agent in self.agents}
        return obs, infos
        
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
            elif actions[agent] == 2:
                x -= 1
            elif actions[agent] == 3:
                x +=1
            elif actions[agent] ==4:
                y+=1
            

            #check collision with kills
            crashed = False
            for building in self.buildings:
                ##if the position of the drone is in the area where the building is
                if building["X_front"] <= x <= building["X_end"] and building["y_front"] <= y <= building["y_end"]:
                    if z <= building["height"]:
                        crashed = True
                        break
            
            
            if (crashed == False):
                reward = 1.0
            else: 
                reward = -0.1
            
            #award for finding the survivor
            if np.array_equal([x,y,z], self.survivor):
                reward = 5.0
                print("survivor found")
                done = True


            done = crashed

            rewards[agent] = reward
            terminations[agent] = done
            truncations[agent] = False
            infos[agent] = {}

            #get the new closest building after moving
            min_dist = float("inf")
            closest_building = None
            for building in self.buildings:
                hx = (building["X_front"] + building["X_end"])/2
                hy = (building["y_front"] + building["y_end"])/2
                dist = np.sqrt((hx - x) ** 2 + (hy-y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_building = building
            
            #new array of observations, the drones position, closest buildings height, and how far away it is from the drone
            obs[agent] = np.array([x,y,z,closest_building["height"], min_dist], dtype=np.float32)


        return obs, rewards, terminations, truncations, infos



                






















