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

        #observations and action spaces - changed to 4 elements
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # up, down, left, right

        #create 2 basic buildings for now -> in the future read in data about the buildings
        # self.buildings = [
        #     building_conditions(5,5,5,5),
        #     building_conditions(12,8,4,3)
        # ]


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        #drone positions
        self.positions = {}
        for agent in self.agents:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0,self.grid_size)

            self.positions[agent] = [x,y]

        #add positions for 2-3 randomly placed survivors
        num_survivors = np.random.randint(2,4)  # 2 or 3 survivors
        self.survivors = []
        for _ in range(num_survivors):
            survivor = [np.random.randint(0,self.grid_size), np.random.randint(0, self.grid_size)]
            self.survivors.append(survivor)
        
        #random buildings
        self.buildings = []
        num_buildings = np.random.randint(2, 6) 
        for _ in range(num_buildings):
            width = np.random.randint(1, 6)
            height = np.random.randint(1, 10)
            x_start = np.random.randint(0, self.grid_size - width)
            y_start = np.random.randint(0, self.grid_size - width)
            building = building_conditions(height, width, x_start, y_start)
            self.buildings.append(building)

        obs = {}
        for agent in self.agents:
            x,y = self.positions[agent]

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

            #closest survivor distance
            survivor_min_dist = float("inf")
            for survivor in self.survivors:
                dist = np.sqrt((survivor[0] - x) ** 2 + (survivor[1] - y) ** 2)
                if dist < survivor_min_dist:
                    survivor_min_dist = dist

            #4-element observation: position + nearest survivor distance + nearest building center distance
            obs[agent] = np.array([x, y, survivor_min_dist, min_dist], dtype=np.float32)
        
        infos = {agent: {} for agent in self.agents}
        return obs, infos
        
    def step(self, actions):
        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            x,y = self.positions[agent]
            if actions[agent] == 0:
                y -=1
            elif actions[agent] ==1:
                y +=1
            elif actions[agent] == 2:
                x -= 1
            elif actions[agent] == 3:
                x +=1

            #keep inside grid
            x = max(0, min(x, self.grid_size-1))
            y = max(0, min(y, self.grid_size-1))

            self.positions[agent] = [x,y]

            #check collision with kills
            crashed = False
            for building in self.buildings:
                ##if the position of the drone is in the area where the building is
                if building["X_front"] <= x <= building["X_end"] and building["y_front"] <= y <= building["y_end"]:
                    crashed = True
                    break
            
            if (crashed == False):
                reward = 1.0
            else: 
                reward = -1.0
            
            #award for finding the survivor and going near survivors
            survivor_found = False
            survivor_min_dist = float("inf")
            survivors_to_remove = []
            
            for i, survivor in enumerate(self.survivors):
                dist = np.sqrt((survivor[0] - x) ** 2 + (survivor[1] - y) ** 2)
                survivor_min_dist = min(survivor_min_dist, dist)
                
                if dist < 1.5:  # found survivor
                    reward += 5.0
                    survivor_found = True
                    survivors_to_remove.append(i)
                elif dist < 3.0:  # near(increase from 1 -3)
                    reward += 2.5 * (1.0 / (dist + 0.1))  # closer -> more reward(1 to .1)
                    #increase reward from 1.5 to 2.5

            
            for i in sorted(survivors_to_remove, reverse=True):
                if i < len(self.survivors):
                    self.survivors.pop(i)

            done = crashed or len(self.survivors) == 0

            rewards[agent] = reward
            terminations[agent] = done
            truncations[agent] = False

            # ✅ update infos to indicate rescued
            infos[agent] = {"rescued": survivor_found}

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
            
            #find new closest survivor distance
            new_survivor_min_dist = float("inf")
            for survivor in self.survivors:
                dist = np.sqrt((survivor[0] - x) ** 2 + (survivor[1] - y) ** 2)
                if dist < new_survivor_min_dist:
                    new_survivor_min_dist = dist
            
            obs[agent] = np.array([x,y,new_survivor_min_dist,min_dist], dtype=np.float32)


        return obs, rewards, terminations, truncations, infos
