"""
Unit tests for PriorityRescueEnv
"""

import pytest
import numpy as np
from gymnasium import spaces
from env.priority_rescue import PriorityRescueEnv


class TestPriorityRescueEnvInitialization:
    """Test environment initialization and basic properties."""

    def test_default_initialization(self):
        """Test environment initializes with default parameters."""
        env = PriorityRescueEnv()
        assert env._num_agents == 3
        assert env.grid_size == 15
        assert env.num_survivors == 6
        assert env.max_steps == 200

    def test_custom_initialization(self):
        """Test environment initializes with custom parameters."""
        env = PriorityRescueEnv(num_agents=5, grid_size=20, num_survivors=10, max_steps=300)
        assert env._num_agents == 5
        assert env.grid_size == 20
        assert env.num_survivors == 10
        assert env.max_steps == 300

    def test_possible_agents_creation(self):
        """Test that possible_agents list is correctly created."""
        env = PriorityRescueEnv(num_agents=4)
        assert len(env.possible_agents) == 4
        assert env.possible_agents == ["drone_0", "drone_1", "drone_2", "drone_3"]

    def test_action_space(self):
        """Test action space is correctly defined."""
        env = PriorityRescueEnv()
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 5  # 4 movements + 1 rescue

    def test_observation_space(self):
        """Test observation space is correctly defined."""
        env = PriorityRescueEnv()
        assert isinstance(env.observation_space, spaces.Box)
        expected_size = 2 + (8 * 5)  # 2 for position + 8 survivors * 5 features
        assert env.observation_space.shape == (expected_size,)
        assert env.observation_space.dtype == np.float32

    def test_urgency_constants(self):
        """Test urgency level constants are defined."""
        assert PriorityRescueEnv.URGENCY_LOW == 0
        assert PriorityRescueEnv.URGENCY_MEDIUM == 1
        assert PriorityRescueEnv.URGENCY_HIGH == 2
        assert PriorityRescueEnv.URGENCY_CRITICAL == 3

    def test_urgency_config(self):
        """Test urgency configurations are properly defined."""
        config = PriorityRescueEnv.URGENCY_CONFIG
        assert len(config) == 4

        # Check critical urgency
        assert config[PriorityRescueEnv.URGENCY_CRITICAL]["health"] == 50
        assert config[PriorityRescueEnv.URGENCY_CRITICAL]["decay"] == 1.0
        assert config[PriorityRescueEnv.URGENCY_CRITICAL]["reward"] == 100

        # Check low urgency
        assert config[PriorityRescueEnv.URGENCY_LOW]["health"] == 150
        assert config[PriorityRescueEnv.URGENCY_LOW]["decay"] == 0.1
        assert config[PriorityRescueEnv.URGENCY_LOW]["reward"] == 10


class TestPriorityRescueEnvReset:
    """Test environment reset functionality."""

    def test_reset_returns_observations_and_infos(self):
        """Test reset returns observations and infos dictionaries."""
        env = PriorityRescueEnv(num_agents=3)
        observations, infos = env.reset()

        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        assert len(observations) == 3
        assert len(infos) == 3

    def test_reset_initializes_agent_positions(self):
        """Test reset initializes agent positions within grid bounds."""
        env = PriorityRescueEnv(num_agents=3, grid_size=10)
        observations, infos = env.reset()

        for agent_id in env.agents:
            pos = env.agent_positions[agent_id]
            assert 0 <= pos[0] < 10
            assert 0 <= pos[1] < 10

    def test_reset_initializes_survivors(self):
        """Test reset creates correct number of survivors with valid data."""
        env = PriorityRescueEnv(num_survivors=5)
        env.reset()

        assert len(env.survivor_data) == 5
        for survivor in env.survivor_data:
            assert "position" in survivor
            assert "health" in survivor
            assert "max_health" in survivor
            assert "urgency" in survivor
            assert "decay_rate" in survivor
            assert "rescue_reward" in survivor
            assert "alive" in survivor
            assert "rescued" in survivor
            assert survivor["alive"] is True
            assert survivor["rescued"] is False

    def test_reset_with_seed_reproducibility(self):
        """Test reset with seed produces reproducible results."""
        env1 = PriorityRescueEnv(num_agents=3, num_survivors=4)
        env2 = PriorityRescueEnv(num_agents=3, num_survivors=4)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Observations should be identical
        for agent_id in obs1:
            np.testing.assert_array_equal(obs1[agent_id], obs2[agent_id])

    def test_reset_clears_counters(self):
        """Test reset clears step counter, rescue count, and death count."""
        env = PriorityRescueEnv()
        env.reset()

        # Simulate some activity
        env.current_step = 50
        env.rescued_count = 3
        env.death_count = 2

        # Reset and verify counters are cleared
        env.reset()
        assert env.current_step == 0
        assert env.rescued_count == 0
        assert env.death_count == 0

    def test_reset_restores_all_agents(self):
        """Test reset restores all possible agents to active state."""
        env = PriorityRescueEnv(num_agents=3)
        env.reset()

        assert len(env.agents) == 3
        assert env.agents == env.possible_agents


class TestPriorityRescueEnvStep:
    """Test environment step functionality."""

    def test_step_returns_correct_tuple(self):
        """Test step returns 5-tuple of dicts."""
        env = PriorityRescueEnv(num_agents=2)
        env.reset()

        actions = {"drone_0": 0, "drone_1": 1}
        result = env.step(actions)

        assert len(result) == 5
        observations, rewards, terminations, truncations, infos = result
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

    def test_step_increments_counter(self):
        """Test step increments the step counter."""
        env = PriorityRescueEnv()
        env.reset()

        initial_step = env.current_step
        env.step({"drone_0": 0, "drone_1": 1, "drone_2": 2})

        assert env.current_step == initial_step + 1

    def test_movement_actions(self):
        """Test movement actions update agent positions correctly."""
        env = PriorityRescueEnv(num_agents=1, grid_size=10)
        env.reset(seed=42)

        # Get initial position
        initial_pos = env.agent_positions["drone_0"].copy()

        # Move up (action 0)
        env.step({"drone_0": 0})
        new_pos = env.agent_positions["drone_0"]

        # Check position changed (unless at boundary)
        if initial_pos[0] > 0:
            assert new_pos[0] == initial_pos[0] - 1
            assert new_pos[1] == initial_pos[1]

    def test_movement_boundary_clipping(self):
        """Test movement is clipped at grid boundaries."""
        env = PriorityRescueEnv(num_agents=1, grid_size=5)
        env.reset()

        # Place agent at top-left corner
        env.agent_positions["drone_0"] = [0, 0]

        # Try to move up and left (should stay at 0,0)
        env.step({"drone_0": 0})  # up
        assert env.agent_positions["drone_0"][0] == 0

        env.step({"drone_0": 2})  # left
        assert env.agent_positions["drone_0"][1] == 0

    def test_step_penalty_for_movement(self):
        """Test that movement actions incur a small penalty."""
        env = PriorityRescueEnv(num_agents=1)
        env.reset()

        _, rewards, _, _, _ = env.step({"drone_0": 0})

        # Movement should have -0.1 penalty (plus decay penalties)
        assert rewards["drone_0"] < 0

    def test_invalid_rescue_penalty(self):
        """Test that rescue action on empty cell gives penalty."""
        env = PriorityRescueEnv(num_agents=1, grid_size=20)
        env.reset(seed=42)

        # Move agent to likely empty position
        env.agent_positions["drone_0"] = [19, 19]

        # Attempt rescue
        _, rewards, _, _, _ = env.step({"drone_0": 4})

        # Should have -1.0 penalty for invalid rescue (plus decay penalties)
        assert rewards["drone_0"] <= -1.0

    def test_successful_rescue(self):
        """Test successful rescue updates survivor state and gives reward."""
        env = PriorityRescueEnv(num_agents=1, num_survivors=1)
        env.reset()

        # Place agent on survivor
        survivor_pos = env.survivor_data[0]["position"]
        env.agent_positions["drone_0"] = survivor_pos

        initial_rescued = env.rescued_count
        expected_reward = env.survivor_data[0]["rescue_reward"]

        # Perform rescue
        _, rewards, _, _, _ = env.step({"drone_0": 4})

        # Verify rescue success
        assert env.rescued_count == initial_rescued + 1
        assert env.survivor_data[0]["rescued"] is True
        assert env.survivor_data[0]["alive"] is False
        assert rewards["drone_0"] >= expected_reward - 1  # Account for small decay penalty

    def test_health_decay(self):
        """Test that survivor health decays over time."""
        env = PriorityRescueEnv(num_survivors=1)
        env.reset(seed=42)

        initial_health = env.survivor_data[0]["health"]
        decay_rate = env.survivor_data[0]["decay_rate"]

        # Take a step
        env.step({"drone_0": 0, "drone_1": 0, "drone_2": 0})

        new_health = env.survivor_data[0]["health"]

        # Health should have decayed
        assert new_health == initial_health - decay_rate

    def test_survivor_death(self):
        """Test that survivor dies when health reaches zero."""
        env = PriorityRescueEnv(num_survivors=1)
        env.reset()

        # Set health below decay rate to ensure death in one step
        decay_rate = env.survivor_data[0]["decay_rate"]
        env.survivor_data[0]["health"] = decay_rate * 0.5  # Less than one decay cycle

        initial_deaths = env.death_count

        # Step until death
        env.step({"drone_0": 0, "drone_1": 0, "drone_2": 0})

        # Survivor should be dead
        assert env.survivor_data[0]["alive"] is False
        assert env.survivor_data[0]["health"] == 0
        assert env.death_count == initial_deaths + 1

    def test_death_penalty(self):
        """Test that survivor death gives large penalty."""
        env = PriorityRescueEnv(num_survivors=1, num_agents=1)
        env.reset()

        # Set health to very low
        env.survivor_data[0]["health"] = 0.5

        # Step and check penalty
        _, rewards, _, _, _ = env.step({"drone_0": 0})

        # Should have -50 penalty for death
        assert rewards["drone_0"] <= -50

    def test_termination_all_survivors_handled(self):
        """Test termination when all survivors are rescued or dead."""
        env = PriorityRescueEnv(num_survivors=2)
        env.reset()

        # Mark all survivors as rescued
        for survivor in env.survivor_data:
            survivor["rescued"] = True
            survivor["alive"] = False

        _, _, terminations, _, _ = env.step({"drone_0": 0, "drone_1": 0, "drone_2": 0})

        # All agents should be terminated
        assert all(terminations.values())

    def test_truncation_max_steps(self):
        """Test truncation when max_steps is reached."""
        env = PriorityRescueEnv(max_steps=5)
        env.reset()

        # Run until max_steps
        for _ in range(4):
            _, _, _, truncations, _ = env.step({"drone_0": 0, "drone_1": 0, "drone_2": 0})
            assert not any(truncations.values())

        # Step 5 should trigger truncation
        _, _, _, truncations, _ = env.step({"drone_0": 0, "drone_1": 0, "drone_2": 0})
        assert all(truncations.values())

    def test_info_contains_required_fields(self):
        """Test that info dict contains required fields."""
        env = PriorityRescueEnv()
        _, infos = env.reset()

        for agent_id in env.agents:
            info = infos[agent_id]
            assert "position" in info
            assert "step" in info
            assert "rescued_count" in info
            assert "death_count" in info
            assert "active_survivors" in info


class TestPriorityRescueEnvObservations:
    """Test observation generation."""

    def test_observation_shape(self):
        """Test observation has correct shape."""
        env = PriorityRescueEnv()
        observations, _ = env.reset()

        expected_shape = (2 + 8 * 5,)
        for obs in observations.values():
            assert obs.shape == expected_shape

    def test_observation_normalization(self):
        """Test observation values are normalized to [0, 1]."""
        env = PriorityRescueEnv()
        observations, _ = env.reset()

        for obs in observations.values():
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_observation_contains_agent_position(self):
        """Test observation contains normalized agent position."""
        env = PriorityRescueEnv(grid_size=10)
        env.reset()

        # Set known position
        env.agent_positions["drone_0"] = [5, 7]

        obs = env._get_observation("drone_0")

        # First two elements should be normalized position
        assert obs[0] == 5 / 10
        assert obs[1] == 7 / 10

    def test_observation_survivor_sorting(self):
        """Test survivors are sorted by urgency then health in observations."""
        env = PriorityRescueEnv(num_survivors=3)
        env.reset()

        # Manually set survivor urgency levels
        env.survivor_data[0]["urgency"] = env.URGENCY_LOW
        env.survivor_data[0]["health"] = 100
        env.survivor_data[1]["urgency"] = env.URGENCY_CRITICAL
        env.survivor_data[1]["health"] = 30
        env.survivor_data[2]["urgency"] = env.URGENCY_CRITICAL
        env.survivor_data[2]["health"] = 20

        env.agent_positions["drone_0"] = [0, 0]

        obs = env._get_observation("drone_0")

        # Extract urgency values (every 5th element starting from index 5)
        urgency_1 = obs[2 + 3]  # First survivor's urgency
        urgency_2 = obs[2 + 5 + 3]  # Second survivor's urgency

        # First survivor should be critical (urgency 3/3 = 1.0)
        assert urgency_1 == 1.0

    def test_observation_excludes_rescued(self):
        """Test rescued survivors are not in observations."""
        env = PriorityRescueEnv(num_survivors=2)
        env.reset()

        # Mark first survivor as rescued
        env.survivor_data[0]["rescued"] = True
        env.survivor_data[0]["alive"] = False

        env.agent_positions["drone_0"] = [0, 0]

        obs = env._get_observation("drone_0")

        # Check that only one survivor is represented
        # Alive status at index 2+4 should be 1.0 (one survivor)
        # Second survivor slot should be zeros
        survivor_2_alive = obs[2 + 5 + 4]  # Second survivor alive status
        assert survivor_2_alive == 0.0  # Should be empty (zero-padded)


class TestPriorityRescueEnvRender:
    """Test environment rendering."""

    def test_render_executes_without_error(self):
        """Test render method executes without raising errors."""
        env = PriorityRescueEnv(grid_size=5)
        env.reset()

        # Should not raise any exceptions
        try:
            env.render()
            assert True
        except Exception as e:
            pytest.fail(f"Render raised exception: {e}")

    def test_render_with_various_states(self):
        """Test render handles different environment states."""
        env = PriorityRescueEnv(num_survivors=2, grid_size=5)
        env.reset()

        # Initial state
        env.render()

        # After some steps
        env.step({"drone_0": 0, "drone_1": 1, "drone_2": 2})
        env.render()

        # With rescued survivor
        env.survivor_data[0]["rescued"] = True
        env.render()

        # With dead survivor
        env.survivor_data[1]["alive"] = False
        env.render()


class TestPriorityRescueEnvEdgeCases:
    """Test edge cases and error handling."""

    def test_single_agent(self):
        """Test environment works with single agent."""
        env = PriorityRescueEnv(num_agents=1)
        observations, _ = env.reset()

        assert len(observations) == 1
        assert "drone_0" in observations

    def test_no_survivors(self):
        """Test environment works with no survivors."""
        env = PriorityRescueEnv(num_survivors=0)
        observations, _ = env.reset()

        assert len(env.survivor_data) == 0
        # Should terminate immediately
        _, _, terminations, _, _ = env.step({"drone_0": 0, "drone_1": 0, "drone_2": 0})
        assert all(terminations.values())

    def test_many_agents(self):
        """Test environment works with many agents."""
        env = PriorityRescueEnv(num_agents=10)
        observations, _ = env.reset()

        assert len(observations) == 10
        assert len(env.possible_agents) == 10

    def test_small_grid(self):
        """Test environment works with small grid."""
        env = PriorityRescueEnv(grid_size=3)
        env.reset()

        # Verify positions are within bounds
        for pos in env.agent_positions.values():
            assert 0 <= pos[0] < 3
            assert 0 <= pos[1] < 3

    def test_large_grid(self):
        """Test environment works with large grid."""
        env = PriorityRescueEnv(grid_size=50)
        env.reset()

        # Verify positions are within bounds
        for pos in env.agent_positions.values():
            assert 0 <= pos[0] < 50
            assert 0 <= pos[1] < 50

    def test_step_with_missing_agent_action(self):
        """Test step handles missing agent actions gracefully."""
        env = PriorityRescueEnv(num_agents=3)
        env.reset()

        # Provide actions for only 2 agents
        actions = {"drone_0": 0, "drone_1": 1}

        # Should not raise error
        try:
            env.step(actions)
            assert True
        except Exception as e:
            pytest.fail(f"Step raised exception with missing action: {e}")

    def test_multiple_rescues_same_step(self):
        """Test multiple agents can rescue different survivors in same step."""
        env = PriorityRescueEnv(num_agents=2, num_survivors=2)
        env.reset()

        # Place each agent on different survivor
        env.agent_positions["drone_0"] = env.survivor_data[0]["position"]
        env.agent_positions["drone_1"] = env.survivor_data[1]["position"]

        # Both attempt rescue
        _, rewards, _, _, _ = env.step({"drone_0": 4, "drone_1": 4})

        # Both should succeed
        assert env.rescued_count == 2
        assert rewards["drone_0"] > 0
        assert rewards["drone_1"] > 0


class TestPriorityRescueEnvRewardShaping:
    """Test reward mechanics and shaping."""

    def test_critical_survivor_higher_reward(self):
        """Test critical survivors give higher rewards than low urgency."""
        critical_config = PriorityRescueEnv.URGENCY_CONFIG[PriorityRescueEnv.URGENCY_CRITICAL]
        low_config = PriorityRescueEnv.URGENCY_CONFIG[PriorityRescueEnv.URGENCY_LOW]

        assert critical_config["reward"] > low_config["reward"]

    def test_critical_survivor_faster_decay(self):
        """Test critical survivors decay faster than low urgency."""
        critical_config = PriorityRescueEnv.URGENCY_CONFIG[PriorityRescueEnv.URGENCY_CRITICAL]
        low_config = PriorityRescueEnv.URGENCY_CONFIG[PriorityRescueEnv.URGENCY_LOW]

        assert critical_config["decay"] > low_config["decay"]

    def test_decay_penalty_accumulates(self):
        """Test that decay penalty is applied each step."""
        env = PriorityRescueEnv(num_agents=1, num_survivors=1)
        env.reset()

        # Take multiple steps without rescuing
        total_penalty = 0
        for _ in range(5):
            _, rewards, _, _, _ = env.step({"drone_0": 0})
            total_penalty += rewards["drone_0"]

        # Should accumulate penalties (movement -0.1 + decay -0.02 per step = -0.12 * 5 = -0.6)
        assert total_penalty < -0.5
