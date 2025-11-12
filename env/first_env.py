import numpy as np
import gymnasium as gym
import osmnx as ox
import shapely
import geopandas as gpd
import pandas as pd
from pettingzoo import ParallelEnv
from gymnasium import spaces
from pyproj import Transformer




class Env(ParallelEnv):
    """
    First Rescue Env
    """
    
    def __init__(self, origin, bbox, num_agents=3):
        # Store parameters (don't use num_agents as it's a PettingZoo property)
        self._num_agents = num_agents
        self.origin = origin
        self.bbox = bbox # bbox in terms of lon/lat
        self.mem_lim = 20

        # Agent names
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:] # copy
                                                                            
        # Each agent gets the coordinates of the entities around it
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.mem_lim,3), dtype=np.float32) # (x,y,type)

        # Each agent can control its velocity and check if there is a survivor near its position
        self.action_space = spaces.Box(low=np.array([-25,-25,0]), high=np.array ([25,25,1]), dtype=np.float32) #(vx,xy,check_flag) 
        
        # get geospatial data and generate map 
        self.generate_map()

    def get_max(self):
        mbbox = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[self.bbox])
        mbbox = mbbox.to_crs(epsg=3857)
        _, _, maxx, maxy = self.lat2met(mbbox).total_bounds
        return maxx,maxy
    
    def generate_map(self):
        # get geospatial data
        water_tags = {
            'natural': ['water', 'coastline', 'beach'],
            'waterway': ['water','river', 'stream', 'canal']
        }
        terrain_tags = {
            'natural': ['peak', 'mountain_range', 'cliff', 'ridge', 'valley']
        }
        park_tags = {
            'leisure': ['park', 'nature_reserve', 'garden'],
            'landuse': ['forest', 'grass', 'meadow']
        }
        # roads = ox.graph_from_polygon(bbox, network_type="drive") 
        # ox.plot_graph(roads)
        # NOTE: not needed since all our agents are air based,
        #       but if there were ground based agents, road networks will help.

        buildings = self.get_features(self.bbox, tags={'building': True}) 
        water = self.get_features(self.bbox, water_tags) 
        parks = self.get_features(self.bbox, park_tags) 
        terrain = self.get_features(self.bbox, terrain_tags)
        
        # setup types for feature classificiation
        water["type"] = "water"
        parks["type"] = "parks"
        buildings["type"] = "building"
        terrain["type"] = "terrain"
    
        # setup origin to be top left corner
        to_meters =  Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        lon,lat = self.origin
        self.origin = to_meters.transform(lon,lat)

        # convert coordinates to meters (long,lat) -> (x,y)m
        water["geometry"] = self.lat2met(water)
        parks["geometry"] = self.lat2met(parks)
        terrain["geometry"] = self.lat2met(terrain)
        buildings["geometry"] = self.lat2met(buildings)
        self.map = gpd.GeoDataFrame(pd.concat([buildings, water, parks, terrain], ignore_index=True))

    def lat2met(self, feat):
        return feat.geometry.apply(lambda geom: shapely.affinity.translate(geom, xoff=-self.origin[0], yoff=-self.origin[1]))

    def get_features(self, bbox, tags):
        try:
            return ox.features_from_polygon(bbox, tags=tags).to_crs(epsg=3857)
        
        except Exception as e:
            print(f"WARNING: This feature does not exist: {tags}")
            return gpd.GeoDataFrame(columns=['id', 'distance', 'feature'], geometry='feature')

    def reset(self, seed=None, options=None):
        """
        Reset the environment and Randomize map (location of surivivors)
        """
        self.agents = self.possible_agents[:]

        # Randomly spread the survivors throughout the map
        mx, my = self.get_max()
        self.num_survivors = np.random.randint(low=1,high=(mx+my)*0.01, dtype = np.int32)
        self.survivor_pos = [
            shapely.geometry.Point(np.random.uniform(0,mx), np.random.uniform(0,my))
            for _ in range(self.num_survivors)
            ]
        self.survivor_pos = gpd.GeoDataFrame(geometry=self.survivor_pos)

        # Distribute positions in the region near the origin (1% of total map size)
        self.pos = {agent: shapely.geometry.Point(np.random.uniform(0,mx*0.01),np.random.uniform(0,my*0.01)) for agent in self.agents}
        
        observations = {}
        for agent in self.agents:        
            observations[agent] = self.get_observations(self.pos[agent], 100)
        
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        print(self.num_survivors)
        """Execute one step."""
        # Simple random rewards and termination
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            self.pos[agent], penalty = self.update_positions(actions[agent], self.pos[agent])
            observations[agent] = self.get_observations(self.pos[agent],100)
            rewards[agent] =  self.get_rewards(self.pos[agent], actions[agent][2]) + penalty 
            terminations[agent] = True if self.num_survivors == 0 else False
            truncations[agent] = False
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos
    
    def get_rewards(self, apos, check):
        x,y = apos.x, apos.y
        check = 1
        if check == 1:
            local_margin=5
            bbox = shapely.box(x-local_margin, y-local_margin, x+local_margin, y+local_margin)
            # Get survivors
            found_survivors = self.get_entities(self.survivor_pos, bbox).geometry
            num_found = len(found_survivors)
            reward  = len(found_survivors)*2 if num_found > 0 else -1
            
            # Remove found survivors from survivor_pos
            if num_found > 0:
                remaining_survivors = self.survivor_pos[~self.survivor_pos.geometry.isin(found_survivors.geometry)]
                self.survivor_pos = remaining_survivors.reset_index(drop=True)
                self.num_survivors -= num_found
            return reward
        
        return 0

    def get_observations(self, apos, field_size):
        # define region
        x,y = apos.x, apos.y
        half = field_size/2
        bbox = shapely.box(x-half, y-half, x+half, y+half)

        # get features in region
        feats = self.get_entities(self.map,bbox)

        # get survivors in region
        survivors = self.get_entities(self.survivor_pos, bbox)

        # get agents in region
        pos = gpd.GeoDataFrame(geometry=list(self.pos.values()))
        agents = self.get_entities(pos,bbox)
        agents = agents[agents.geometry.distance(apos) > 1e-6] # exclude current agent pos

        # Combine all into a numpy array

        feats = self.get_coords(feats, 0)
        survivors = self.get_coords(survivors, 1)
        agents = self.get_coords(agents, 2)

        all_entities = np.vstack([feats, survivors, agents]) # combine everything

        #truncate/pad if required
        ae = all_entities[:self.mem_lim, :] 
        ae = self.pad(ae)
        return ae 
    
    def pad(self, ae):
        if len(ae) < self.mem_lim:
            pad = np.ones((self.mem_lim - ae.shape[0], ae.shape[1]), dtype=ae.dtype) * -1
            return np.vstack([ae,pad])
        return ae

    
    def get_coords(self, gdf, type_id):
            if gdf.empty:  # or if len(gdf) == 0
                return np.empty((0, 3), dtype=np.float32)
             
            return np.array([[geom.centroid.x, geom.centroid.y, type_id] for geom in gdf.geometry], dtype=np.float32)
    
    def get_entities(self, entity, bbox):
        possible_matches_index = list(entity.sindex.intersection(bbox.bounds))
        possible_matches = entity.iloc[possible_matches_index]
        return possible_matches[possible_matches.intersects(bbox)]
    
    def in_bounds(self, x,y,vx,vy):
        maxx, maxy = self.get_max()
        if x + vx <= 0 or x + vx >= maxx or y + vy <= 0 or y + vy >= maxy:
            return False
        return True

    def update_positions(self, action, apos):
        """
        update positions based on actions
        returns (x_new, y_new), penalty
        penalty is given if the agent tries to go out of bounds
        """ 
        x,y = apos.x, apos.y
        vx,vy, _ = action

        # update positions 
        if self.in_bounds(x,y, vx,vy):
            x += vx
            y += vy
            return shapely.geometry.Point(x,y), 0

        else:
            return shapely.geometry.Point(x,y), -1

        

