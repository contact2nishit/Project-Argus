import numpy as np # type: ignore
import gymnasium as gym # type: ignore
import osmnx as ox # type: ignore
import shapely # type: ignore
import geopandas as gpd # type: ignore
import pandas as pd # type: ignore
import copy # type: ignore

from gymnasium import spaces # type: ignore
from pyproj import Transformer # type: ignore
from pettingzoo import ParallelEnv #type: ignore

class Env(ParallelEnv):
    """
    First Rescue Env
    """
    
    def __init__(self, origin, bbox, num_agents=5,battery_capacity=100, mem_limit=5):
        # Store parameters (don't use num_agents as it's a PettingZoo property)
        self._num_agents = num_agents
        self.origin = origin
        self.bbox = bbox # bbox in terms of lon/lat
        self.mem_lim = mem_limit
        self.battery_capacity = battery_capacity

        # Agent names
        self.possible_agents = [i for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Agent battery
        self._battery = {}
        for agent in self.agents:
            self._battery[agent] = battery_capacity

        # Each agent gets the coordinates of the entities around it
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.mem_lim,5), dtype=np.float32) # (battery_level, [relative x, relative y, distance, type] )
            for agent in self.possible_agents
        }
        
        # Each agent can control its velocity and check if there is a survivor near its position
        self.action_spaces = {
            agent: spaces.Box(low=np.array([-5,-5,0]), high=np.array ([5,5,1]), dtype=np.float32) 
            for agent in self.possible_agents #(vx,xy,check_flag) 
        }
        
        # get geospatial data and generate map 
        self.generate_map()

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
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
    
    def get_max(self):
        mbbox = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[self.bbox])
        mbbox = mbbox.to_crs(epsg=3857)
        _, _, maxx, maxy = self.lat2met(mbbox).total_bounds
        return maxx,maxy

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
        self.battery = copy.deepcopy(self._battery)
        self.done = False

        # Randomly spread the survivors throughout the map
        mx, my = self.get_max()
        self.num_survivors = np.random.randint(low=(mx+my)*0.05,high=(mx+my)*0.3, dtype = np.int32)
        self.survivor_pos = [
            shapely.geometry.Point(np.random.uniform(0,mx), np.random.uniform(0,my))
            for _ in range(self.num_survivors)
            ]
        # Distribute agent positions in the region near the origin (1% of total map size)
        self.pos =  [
            shapely.geometry.Point(np.random.uniform(0,mx*0.01),np.random.uniform(0,my*0.01))
            for _ in self.agents
        ]
        
        infos =  {}
        observations = {}
        for idx, agent in enumerate(self.agents):
            observations[agent] = self.get_observations(self.pos[idx], field_size=100)
            infos[agent] = {}
        

        return observations, infos
    
    def step(self, actions):
        """Execute one step."""
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        converted_actions = {}

        for agent, act in actions.items():
            # torch tensor
            if hasattr(act, "detach"):
                converted_actions[agent] = act.detach().cpu().numpy()
            # numpy array
            elif hasattr(act, "astype"):
                converted_actions[agent] = np.array(act, dtype=np.float32)
            # list or float/int
            else:
                converted_actions[agent] = np.asarray(act, dtype=np.float32)

        # mx, my = self.get_max()
        # vx,vy,_ = action
        # self.done = False
        # print(f"SURVIVORS={self.num_survivors}, x={self.pos.x:.1f}, y={self.pos.y:.1f}, vx={vx:.1f}, vy={vy:.1f}, bat={self.battery_capacity:.1f}, mX={mx}, mY={my}, dis={self.get_distance(self.pos)}")

        for idx, agent in enumerate(self.agents):
            self.pos[agent] = self.update_positions(converted_actions[idx], self.pos[idx])
            observations[agent] = self.get_observations(self.pos[idx], field_size=100)
            rewards[agent] =  self.get_rewards(self.pos[idx], converted_actions[idx]) 
            terminations[agent] = True if self.num_survivors <= 0 else False
            truncations[agent] = True if self.battery[idx] <= 0 else False
            infos[agent] = {}
        
        return observations, rewards, terminations, truncations, infos
    
    def get_rewards(self, apos, action):
        x, y = apos.x, apos.y
        vx, vy, check = action
        check = round(check)
        maxx, maxy = self.get_max()

        # predict next position (but we will clip)
        next_x = x + vx
        next_y = y + vy

        # clip next position (so agent cannot escape)
        clipped_x = np.clip(next_x, 0.0, maxx)
        clipped_y = np.clip(next_y, 0.0, maxy)

        oob = (next_x != clipped_x) or (next_y != clipped_y)

        # distance to nearest survivor BEFORE and AFTER action
        # if there are no survivors left -> small positive reward to finish
        if self.num_survivors <= 0:
            return 0.1

        nearest = self.survivor_pos[0]
        
        sx, sy = nearest.x, nearest.y
        dist_before = np.hypot(sx - x, sy - y)
        dist_after = np.hypot(sx - clipped_x, sy - clipped_y)

        # compute approximate maximum possible diagonal distance on the map for normalization
        maxx, maxy = self.get_max()
        max_dist = np.hypot(maxx, maxy) + 1e-8

        # proximity reward: normalized improvement toward survivor -> positive if we get closer, negative if we get farther
        proximity_reward = (dist_before - dist_after) / (max_dist + 1e-8)

        # small step/time penalty to encourage efficiency
        time_penalty = -0.01

        # soft OOB penalty (small) only if we tried to go out
        oob_penalty = -0.5 if oob else 0.0

        # check action: if check==1, reward for nearby survivors
        reward = proximity_reward + time_penalty + oob_penalty
        if dist_after < 10:
            self.done = True
            reward += 1

        if check == 1:
            local_margin = 10
            bbox = shapely.box(x - local_margin, y - local_margin, x + local_margin, y + local_margin)
            survivor_pos = gpd.GeoDataFrame(
                pd.DataFrame({'geometry': pd.Series(self.survivor_pos)}), 
                geometry='geometry', crs="EPSG:3857"
            )
            found = self.get_entities(survivor_pos, bbox).geometry
            num_found = len(found)
            if num_found > 0:
                # big positive reward for actually finding survivors
                reward += num_found * 10 
                # remove survivors from global list (persist)
                remaining = survivor_pos[~survivor_pos.geometry.isin(found)]
                self.survivor_pos = list(remaining.geometry.values)
                self.num_survivors -= num_found
                
        return float(reward)

    def get_observations(self, apos, field_size):
        # define region
        x,y = apos.x, apos.y
        half = field_size/2
        mx, my = self.get_max()
        bbox = shapely.box(x-half, y-half, x+half, y+half)

        # get features in region
        feats = self.get_entities(self.map,bbox)

        # Get max area
        self.map['area'] = self.map.geometry.area  # GeoPandas computes area in the CRS units
        max_area = self.map['area'].max()

        # get survivors in region
        survivor_pos = gpd.GeoDataFrame(pd.DataFrame({'geometry': pd.Series(self.survivor_pos)}), geometry='geometry', crs="EPSG:3857")        
        survivors = self.get_entities(survivor_pos, bbox)
        survivors["type"] = "survivor"

        # get agents in region
        pos = gpd.GeoDataFrame(geometry=self.pos, crs="EPSG:3857")
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

        obs = np.zeros((self.mem_lim, 4), dtype=np.float32)
        for i, row in ae.iterrows():
            geom = row.geometry
            if geom.is_empty:
                obs[i] = [-1, -1, 1, -1,]  # Sentinel values for padding
                continue

            # Nearest point on geometry
            nearest_point = shapely.ops.nearest_points(apos, geom)[1]
            
            obs[i] = [
                (nearest_point.x - apos.x)/mx,              # relative x to nearest point
                (nearest_point.y - apos.y)/my,              # relative y to nearest point 
                geom.area/max_area,                                  # size indicator
                self.type2id(row.get('type','unkown'))/6      # type           
            ]   
        battery_column = np.full((self.mem_lim, 1), self.battery_capacity, dtype=np.float32)
        return np.hstack([obs, battery_column])

    def type2id(self, type):
        t2id = {'water': 1, 'parks': 2, 'building': 3, 'terrain': 4, 'survivor': 5, 'agent': 6}
        return t2id.get(type, -1)
        
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
            print("going out of map bounds")
            return False
        
        if len(self.get_entities(self.buildings, shapely.geometry.Point(x + vx, y + vy))) > 0: #building bounds
            print("building bounds")
            return False
        
        return True

    def update_positions(self, actions, apos):
        """
        update positions and battery based on actions
        returns (x_new, y_new), penalty
        penalty is given if the agent tries to go out of bounds
        """ 

        x,y = apos.x, apos.y
        vx,vy, _ = actions
        maxx, maxy = self.get_max()
        nx = np.clip(x + vx,0, maxx)
        ny = np.clip(y + vy,0, maxy)

        self.battery_capacity = self.update_battery(self.battery_capacity,x,y)

        return shapely.geometry.Point(nx, ny)
    
    def update_battery(self, battery, x,y):
        if len(self.get_entities(self.water, shapely.geometry.Point(x, y))) > 0:
            battery -= 1.5

        elif len(self.get_entities(self.parks, shapely.geometry.Point(x, y))) > 0:
            battery -= 1.5

        elif len(self.get_entities(self.terrain, shapely.geometry.Point(x, y))) > 0:
            battery -= -1.2

        else:   
            battery -= 1.1

        return battery