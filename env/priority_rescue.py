"""
Priority-Based Rescue Environment with Time Decay

This simulation represents a multi-agent environment where survivors have different urgency levels
and a health attribute that decays over time. Drones must prioritize critical survivors
while managing time constraints.
"""

import numpy as np


from gymnasium import spaces
from pettingzoo import ParallelEnv


class PriorityRescueEnv(ParallelEnv):
    """
    Priority-based rescue environment with time decay mechanics.

    Survivors have urgency levels (LOW, MEDIUM, HIGH, CRITICAL) that determine:
    - Initial health
    - Health decay rate
    - Rescue reward value

    Drones must learn to triage and prioritize critical survivors.
    """

    # Urgency level constants
    URGENCY_CRITICAL = 3
    URGENCY_HIGH = 2
    URGENCY_MEDIUM = 1
    URGENCY_LOW = 0

    # Urgency configurations: (initial_health, decay_rate, rescue_reward)
    URGENCY_CONFIG = {
        URGENCY_CRITICAL: {"health": 50, "decay": 1.0, "reward": 100},
        URGENCY_HIGH:     {"health": 75, "decay": 0.5, "reward": 50},
        URGENCY_MEDIUM:   {"health": 100, "decay": 0.25, "reward": 25},
        URGENCY_LOW:      {"health": 150, "decay": 0.1, "reward": 10},
    }

    def __init__(self, num_agents=3, grid_size=15, num_survivors=6, max_steps=200):
        """
        Initialize the Priority Rescue Environment.

        Args:
            num_agents: Number of drone agents (default: 3)
            grid_size: Size of the square grid (default: 15)
            num_survivors: Number of survivors to spawn (default: 6)
            max_steps: Maximum steps before episode truncation (default: 200)
        """
        self._num_agents = num_agents
        self.grid_size = grid_size
        self.num_survivors = num_survivors
        self.max_steps = max_steps

        # PettingZoo required attributes
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=rescue
        self.action_space = spaces.Discrete(5)

        # Observation space: 2 (own pos) + 8 survivors * 5 features = 42 features
        # Features per survivor: [rel_x, rel_y, health, urgency, alive]
        self.max_observable_survivors = 8
        obs_size = 2 + (self.max_observable_survivors * 5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # State tracking
        self.agent_positions = {}
        self.survivor_data = []  # List of dicts representing survivors with position, health, urgency, alive
        self.current_step = 0
        self.rescued_count = 0
        self.death_count = 0

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.rescued_count = 0
        self.death_count = 0

        # Initialize agent positions randomly
        self.agent_positions = {}
        for agent in self.agents:
            self.agent_positions[agent] = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ]

        # Initialize survivors with random urgency levels
        self.survivor_data = []
        for _ in range(self.num_survivors):
            # Random urgency level (weighted towards less critical)
            urgency = np.random.choice(
                [self.URGENCY_LOW, self.URGENCY_MEDIUM, self.URGENCY_HIGH, self.URGENCY_CRITICAL],
                p=[0.4, 0.3, 0.2, 0.1]  # More low urgency, fewer critical
            )
            config = self.URGENCY_CONFIG[urgency]

            # Define survivor metrics
            survivor = {
                "position": [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ],
                "health": float(config["health"]),
                "max_health": float(config["health"]),
                "urgency": urgency,
                "decay_rate": config["decay"],
                "rescue_reward": config["reward"],
                "alive": True,
                "rescued": False
            }
            self.survivor_data.append(survivor)

        # Generate initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """Execute one environment step."""
        self.current_step += 1

        # Track rewards for each agent
        rewards = {agent: 0.0 for agent in self.agents}

        # Process movement and rescue actions
        for agent in self.agents:
            if agent not in actions:
                continue

            action = actions[agent]

            # Movement actions (0-3)
            if action < 4:
                self._move_agent(agent, action)
                rewards[agent] -= 0.1  # Small step penalty for efficiency

            # Rescue action (4)
            elif action == 4:
                rescue_reward = self._attempt_rescue(agent)
                rewards[agent] += rescue_reward

        # Decay survivor health and check for deaths
        for survivor in self.survivor_data:
            if survivor["alive"] and not survivor["rescued"]:
                # Decay health
                survivor["health"] -= survivor["decay_rate"]

                # Apply small penalty for ongoing health decay (creates urgency)
                for agent in self.agents:
                    rewards[agent] -= 0.02

                # Check if survivor died
                if survivor["health"] <= 0:
                    survivor["health"] = 0
                    survivor["alive"] = False
                    self.death_count += 1
                    # Big penalty for letting survivor die
                    for agent in self.agents:
                        rewards[agent] -= 50

        # Generate observations and check termination
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        # Check termination conditions
        all_done = all(not s["alive"] or s["rescued"] for s in self.survivor_data)
        truncated = self.current_step >= self.max_steps

        terminations = {agent: all_done for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _move_agent(self, agent, action):
        """Move agent based on action (0=up, 1=down, 2=left, 3=right)."""
        action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        delta = action_map[action]
        current_pos = self.agent_positions[agent]
        new_pos = [
            current_pos[0] + delta[0],
            current_pos[1] + delta[1]
        ]

        # Keep within bounds
        new_pos[0] = max(0, min(self.grid_size - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.grid_size - 1, new_pos[1]))

        self.agent_positions[agent] = new_pos

    def _attempt_rescue(self, agent):
        """Attempt to rescue survivor at agent's current position."""
        agent_pos = self.agent_positions[agent]

        # Check if there's a survivor at this position
        for survivor in self.survivor_data:
            if (survivor["alive"] and
                not survivor["rescued"] and
                survivor["position"] == agent_pos):

                # Successful rescue!
                survivor["rescued"] = True
                survivor["alive"] = False  # No longer needs rescue
                self.rescued_count += 1

                return survivor["rescue_reward"]

        # Invalid rescue action (no survivor here)
        return -1.0

    def _get_observation(self, agent):
        """
        Generate observation for an agent.

        Observation includes:
        - Agent's own normalized position (2 features)
        - For each survivor (up to max_observable_survivors):
          - Relative position (2 features)
          - Normalized health (1 feature)
          - Normalized urgency level (1 feature)
          - Alive status (1 feature)
        """
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Own position (normalized)
        agent_pos = self.agent_positions[agent]
        obs[0] = agent_pos[0] / self.grid_size
        obs[1] = agent_pos[1] / self.grid_size

        # Survivor information (sorted by urgency, then health)
        active_survivors = [
            s for s in self.survivor_data
            if s["alive"] and not s["rescued"]
        ]
        # Sort by urgency (descending), then by health (ascending)
        active_survivors.sort(key=lambda s: (-s["urgency"], s["health"]))

        # Fill observation with survivor data (up to max_observable_survivors)
        for i, survivor in enumerate(active_survivors[:self.max_observable_survivors]):
            base_idx = 2 + (i * 5)

            # Relative position (normalized)
            rel_x = (survivor["position"][0] - agent_pos[0]) / self.grid_size
            rel_y = (survivor["position"][1] - agent_pos[1]) / self.grid_size
            obs[base_idx] = (rel_x + 1) / 2  # Normalize to [0, 1]
            obs[base_idx + 1] = (rel_y + 1) / 2

            # Normalized health
            obs[base_idx + 2] = survivor["health"] / survivor["max_health"]

            # Normalized urgency level
            obs[base_idx + 3] = survivor["urgency"] / 3.0  # 0-3 -> 0-1

            # Alive status
            obs[base_idx + 4] = 1.0 if survivor["alive"] else 0.0

        return obs

    def _get_info(self, agent):
        """Get info dictionary for an agent."""
        return {
            "position": self.agent_positions[agent],
            "step": self.current_step,
            "rescued_count": self.rescued_count,
            "death_count": self.death_count,
            "active_survivors": sum(
                1 for s in self.survivor_data
                if s["alive"] and not s["rescued"]
            )
        }

    def render(self):
        """
        Render the environment state to console.

        Legend:
        . = empty cell
        D = drone
        C = critical survivor (red)
        H = high urgency survivor (orange)
        M = medium urgency survivor (yellow)
        L = low urgency survivor (green)
        X = dead survivor
        """
        # Create empty grid
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place survivors
        urgency_symbols = {
            self.URGENCY_CRITICAL: "C",
            self.URGENCY_HIGH: "H",
            self.URGENCY_MEDIUM: "M",
            self.URGENCY_LOW: "L"
        }

        for survivor in self.survivor_data:
            pos = survivor["position"]
            if survivor["rescued"]:
                continue  # Don't show rescued survivors
            elif not survivor["alive"]:
                grid[pos[0]][pos[1]] = "X"
            else:
                symbol = urgency_symbols[survivor["urgency"]]
                # Show health percentage
                health_pct = int((survivor["health"] / survivor["max_health"]) * 100)
                grid[pos[0]][pos[1]] = f"{symbol}{health_pct:02d}"

        # Place drones (with ID number)
        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            grid[pos[0]][pos[1]] = f"D{i}"

        # Print grid
        print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
        print(f"Rescued: {self.rescued_count}/{self.num_survivors} | Deaths: {self.death_count}")
        print("   " + " ".join(f"{i:4d}" for i in range(self.grid_size)))
        for i, row in enumerate(grid):
            print(f"{i:2d} " + " ".join(f"{cell:>4s}" for cell in row))
        print("\nLegend: D=Drone, C=Critical, H=High, M=Medium, L=Low, X=Dead")
        print("Numbers after urgency letter = health percentage")
