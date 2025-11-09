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
    
    def __init__(self, origin, bbox, num_agents=1,battery_capacity=500):
        # Store parameters (don't use num_agents as it's a PettingZoo property)
        self._num_agents = num_agents
        self.origin = origin
        self.bbox = bbox # bbox in terms of lon/lat
        self.mem_lim = 20
        self.battery_capacity = battery_capacity

        # Agent names
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:] # copy
                                                                            
        # Each agent gets the coordinates of the entities around it
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.mem_lim,2), dtype=np.float32) # (battery_level, [relative x, relative y, distance, area, type] )

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
    
        # setup origin to be bottom left corner
        to_meters =  Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        lon,lat = self.origin
        self.origin = to_meters.transform(lon,lat)

        # convert coordinates to meters (long,lat) -> (x,y)m
        water["geometry"] = self.lat2met(water)
        parks["geometry"] = self.lat2met(parks)
        terrain["geometry"] = self.lat2met(terrain)
        buildings["geometry"] = self.lat2met(buildings)
        self.map = gpd.GeoDataFrame(pd.concat([buildings, water, parks, terrain], ignore_index=True))

        self.water = water
        self.buildings = buildings
        self.parks = parks
        self.terrain = terrain

    def lat2met(self, feat):
        return feat.geometry.apply(lambda geom: shapely.affinity.translate(geom, xoff=-self.origin[0], yoff=-self.origin[1]))

    def get_features(self, bbox, tags):
        try:
            return ox.features_from_polygon(bbox, tags=tags).to_crs(epsg=3857)
        
        except Exception as e:
            print(f"WARNING: This feature does not exist: {tags}")
            return gpd.GeoDataFrame(columns=['id', 'distance', 'feature'], geometry='feature', crs="EPSG:3857")

    def reset(self, seed=None, options=None):
        """
        Reset the environment and Randomize map (location of surivivors)
        """
        self.agents = self.possible_agents[:]

        # Randomly spread the survivors throughout the map
        mx, my = self.get_max()
        self.num_survivors = np.random.randint(low=max(mx,my)*0.01,high=(mx+my)*0.01, dtype = np.int32)
        self.survivor_pos = [
            shapely.geometry.Point(np.random.uniform(0,mx), np.random.uniform(0,my))
            for _ in range(self.num_survivors)
            ]
        self.survivor_pos = gpd.GeoDataFrame(geometry=self.survivor_pos, crs="EPSG:3857")

        # Distribute agent positions in the region near the origin (1% of total map size)
        self.pos = {agent: shapely.geometry.Point(np.random.uniform(0,mx*0.01),np.random.uniform(0,my*0.01)) for agent in self.agents}
        self.battery = {agent: self.battery_capacity for agent in self.agents}
        
        observations = {}
        for agent in self.agents:        
            observations[agent] = self.get_observations(self.pos[agent], field_size=100, agent=agent)
        
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        print(f"    num survivors: {self.num_survivors}")
        """Execute one step."""
        # Simple random rewards and termination
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            self.pos[agent], penalty = self.update_positions(actions, agent)
            observations[agent] = self.get_observations(self.pos[agent], field_size=100, agent=agent)
            rewards[agent] =  self.get_rewards(self.pos[agent], actions[agent][2]) + penalty 
            terminations[agent] = True if self.num_survivors == 0 else False
            truncations[agent] = True if self.battery[agent] <= 0 else False
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos
    
    def get_rewards(self, apos, check):
        x,y = apos.x, apos.y
        check = round(check)
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
        
        return -0.2

    def get_observations(self, apos, field_size, agent):
        # define region
        x,y = apos.x, apos.y
        half = field_size/2
        bbox = shapely.box(x-half, y-half, x+half, y+half)

        # get features in region
        feats = self.get_entities(self.map,bbox)

        # get survivors in region
        survivors = self.get_entities(self.survivor_pos, bbox)
        survivors["type"] = "survivor"

        # get agents in region
        pos = gpd.GeoDataFrame(geometry=list(self.pos.values()), crs="EPSG:3857")
        agents = self.get_entities(pos,bbox)
        agents = agents[agents.geometry.distance(apos) > 1e-6] # exclude current agent pos
        agents["type"] = "agent"

        # Combine all
        all_entities = gpd.GeoDataFrame(
            pd.concat([feats, survivors, agents], ignore_index=True), 
            crs="EPSG:3857"
        )

        #truncate/pad if required
        ae = all_entities.iloc[:self.mem_lim]
        ae = self.pad(ae)

        #sort by distance

        obs = np.zeros((self.mem_lim, 5), dtype=np.float32)
        for i, row in ae.iterrows():
            geom = row.geometry
            if geom.is_empty:
                obs[i] = [-1, -1, -1, -1, -1]  # Sentinel values for padding
                continue

            # Nearest point on geometry
            nearest_point = shapely.ops.nearest_points(apos, geom)[1]
            
            obs[i] = [
                nearest_point.x - apos.x,  # relative x to nearest point
                nearest_point.y - apos.y,  # relative y to nearest point
                apos.distance(geom),       # distance to geometry
                geom.area,                 # size indicator
                self.type2id(row.type)                
            ]
        return [self.battery[agent], obs]

    def type2id(self, type):
        if type == "water":
            return 1
        if type == "parks":
            return 2
        if type == "building":
            return 3
        if type == "terrain":
            return 4
        if type == "survivor":
            return 5
        if type == "agent":
            return 6
        else:
            return -1
        
    def pad(self, ae):
        if len(ae) < self.mem_lim:
            pad_len = self.mem_lim - len(ae)
            pad_df = gpd.GeoDataFrame(
                {col: [-1]*pad_len for col in ae.columns if col != 'geometry'},
                geometry=[shapely.geometry.Point()] * pad_len,
                crs="EPSG:3857"
            )
            ae = pd.concat([ae, pad_df], ignore_index=True)
        return ae

    
    def get_entities(self, entity, bbox):
        possible_matches_index = list(entity.sindex.intersection(bbox.bounds))
        possible_matches = entity.iloc[possible_matches_index]
        return possible_matches[possible_matches.intersects(bbox)]
    
    def in_bounds(self, x,y,vx,vy):
        maxx, maxy = self.get_max()
        if x + vx <= 0 or x + vx >= maxx or y + vy <= 0 or y + vy >= maxy: #map bounds
            return False
        
        if len(self.get_entities(self.buildings, shapely.geometry.Point(x + vx, y + vy))) > 0: #building bounds
            return False
        
        return True

    def update_positions(self, actions, agent):
        """
        update positions and battery based on actions
        returns (x_new, y_new), penalty
        penalty is given if the agent tries to go out of bounds
        """ 
        action = actions[agent]
        apos = self.pos[agent]
        x,y = apos.x, apos.y
        vx,vy, _ = action

        # update positions and battery
        if self.in_bounds(x,y, vx,vy):
            x += vx
            y += vy
            self.battery[agent] = self.update_battery(self.battery[agent],x,y,vx,vy)
            return shapely.geometry.Point(x,y), 0

        else:
            return shapely.geometry.Point(x,y), -1
    
    def update_battery(self, battery, x,y,vx,vy):
        if len(self.get_entities(self.water, shapely.geometry.Point(x, y))) > 0:
            battery -= (vx + vy)*0.3

        elif len(self.get_entities(self.parks, shapely.geometry.Point(x, y))) > 0:
            battery -= (vx + vy)*0.4

        elif len(self.get_entities(self.terrain, shapely.geometry.Point(x, y))) > 0:
            battery -= (vx + vy)*0.6

        else:   
            battery -= (vx + vy)*0.1

        return battery