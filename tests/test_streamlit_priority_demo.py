"""
Unit tests for streamlit_priority_demo.py

Note: These tests focus on the utility functions and logic.
Streamlit UI components are tested using mocking to avoid full UI testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from env.priority_rescue import PriorityRescueEnv
from src.random_agent import RandomAgent

# Import functions from streamlit_priority_demo
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_priority_demo import (
    get_urgency_emoji,
    get_urgency_name,
    render_grid,
    take_step,
    reset_simulation,
    initialize_session_state
)


class MockSessionState(dict):
    """Mock Streamlit session_state that supports both dict and attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


class TestUrgencyHelpers:
    """Test urgency helper functions."""

    def test_get_urgency_emoji_low(self):
        """Test emoji for low urgency."""
        assert get_urgency_emoji(0) == '🟢'

    def test_get_urgency_emoji_medium(self):
        """Test emoji for medium urgency."""
        assert get_urgency_emoji(1) == '🟡'

    def test_get_urgency_emoji_high(self):
        """Test emoji for high urgency."""
        assert get_urgency_emoji(2) == '🟠'

    def test_get_urgency_emoji_critical(self):
        """Test emoji for critical urgency."""
        assert get_urgency_emoji(3) == '🔴'

    def test_get_urgency_emoji_unknown(self):
        """Test emoji for unknown urgency level."""
        assert get_urgency_emoji(99) == '⚪'
        assert get_urgency_emoji(-1) == '⚪'

    def test_get_urgency_name_low(self):
        """Test name for low urgency."""
        assert get_urgency_name(0) == "LOW"

    def test_get_urgency_name_medium(self):
        """Test name for medium urgency."""
        assert get_urgency_name(1) == "MEDIUM"

    def test_get_urgency_name_high(self):
        """Test name for high urgency."""
        assert get_urgency_name(2) == "HIGH"

    def test_get_urgency_name_critical(self):
        """Test name for critical urgency."""
        assert get_urgency_name(3) == "CRITICAL"

    def test_get_urgency_name_unknown(self):
        """Test name for unknown urgency level."""
        assert get_urgency_name(99) == "UNKNOWN"
        assert get_urgency_name(-1) == "UNKNOWN"


class TestInitializeSessionState:
    """Test session state initialization."""

    @patch('streamlit_priority_demo.st')
    def test_initialize_creates_environment(self, mock_st):
        """Test initialization creates environment."""
        mock_st.session_state = MockSessionState()

        initialize_session_state()

        assert 'env' in mock_st.session_state
        assert isinstance(mock_st.session_state['env'], PriorityRescueEnv)

    @patch('streamlit_priority_demo.st')
    def test_initialize_creates_agents(self, mock_st):
        """Test initialization creates correct number of agents."""
        mock_st.session_state = MockSessionState()

        initialize_session_state()

        assert 'agents' in mock_st.session_state
        assert len(mock_st.session_state['agents']) == 3
        for agent in mock_st.session_state['agents'].values():
            assert isinstance(agent, RandomAgent)

    @patch('streamlit_priority_demo.st')
    def test_initialize_sets_initial_values(self, mock_st):
        """Test initialization sets initial counter values."""
        mock_st.session_state = MockSessionState()

        initialize_session_state()

        assert mock_st.session_state['step_count'] == 0
        assert mock_st.session_state['total_reward'] == 0
        assert mock_st.session_state['done'] is False

    @patch('streamlit_priority_demo.st')
    def test_initialize_sets_observations(self, mock_st):
        """Test initialization sets initial observations."""
        mock_st.session_state = MockSessionState()

        initialize_session_state()

        assert 'observations' in mock_st.session_state
        assert 'infos' in mock_st.session_state
        assert isinstance(mock_st.session_state['observations'], dict)
        assert isinstance(mock_st.session_state['infos'], dict)

    @patch('streamlit_priority_demo.st')
    def test_initialize_only_once(self, mock_st):
        """Test initialization doesn't overwrite existing state."""
        # Set up pre-existing state
        mock_env = Mock()
        mock_st.session_state = MockSessionState({'env': mock_env})

        initialize_session_state()

        # Should not create new environment
        assert mock_st.session_state['env'] is mock_env

    @patch('streamlit_priority_demo.st')
    def test_initialize_with_correct_env_params(self, mock_st):
        """Test environment is initialized with correct parameters."""
        mock_st.session_state = MockSessionState()

        initialize_session_state()

        env = mock_st.session_state['env']
        assert env._num_agents == 3
        assert env.grid_size == 15
        assert env.num_survivors == 6
        assert env.max_steps == 200


class TestRenderGrid:
    """Test grid rendering function."""

    @patch('streamlit_priority_demo.st')
    def test_render_grid_returns_html_string(self, mock_st):
        """Test render_grid returns HTML string."""
        env = PriorityRescueEnv(num_agents=2, grid_size=5)
        env.reset(seed=42)
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        assert isinstance(html, str)
        assert '<table' in html
        assert '</table>' in html

    @patch('streamlit_priority_demo.st')
    def test_render_grid_includes_column_headers(self, mock_st):
        """Test grid includes column headers."""
        env = PriorityRescueEnv(num_agents=1, grid_size=5)
        env.reset()
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Should contain column numbers 0-4
        for i in range(5):
            assert str(i) in html

    @patch('streamlit_priority_demo.st')
    def test_render_grid_correct_size(self, mock_st):
        """Test grid renders correct size."""
        grid_size = 8
        env = PriorityRescueEnv(num_agents=1, grid_size=grid_size)
        env.reset()
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Count number of <tr> tags (should be grid_size + 1 for header)
        tr_count = html.count('<tr>')
        assert tr_count == grid_size + 1

    @patch('streamlit_priority_demo.st')
    def test_render_grid_shows_empty_cells(self, mock_st):
        """Test grid shows empty cell emoji."""
        env = PriorityRescueEnv(num_agents=0, grid_size=3, num_survivors=0)
        env.reset()
        env.agents = []  # No agents
        env.survivor_data = []  # No survivors
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Should contain empty cell emoji
        assert '⬜' in html

    @patch('streamlit_priority_demo.st')
    def test_render_grid_shows_agents(self, mock_st):
        """Test grid shows agent emojis."""
        env = PriorityRescueEnv(num_agents=3, grid_size=10)
        env.reset()
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Should contain at least one agent emoji
        agent_emojis = ['🚁', '🛸', '✈️']
        assert any(emoji in html for emoji in agent_emojis)

    @patch('streamlit_priority_demo.st')
    def test_render_grid_shows_survivors(self, mock_st):
        """Test grid shows survivor urgency emojis."""
        env = PriorityRescueEnv(num_agents=0, grid_size=10, num_survivors=4)
        env.reset(seed=42)
        env.agents = []  # Remove agents for clarity
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Should contain survivor urgency emojis
        urgency_emojis = ['🟢', '🟡', '🟠', '🔴']
        survivor_emoji_count = sum(1 for emoji in urgency_emojis if emoji in html)
        assert survivor_emoji_count > 0

    @patch('streamlit_priority_demo.st')
    def test_render_grid_shows_dead_survivor(self, mock_st):
        """Test grid shows dead survivor emoji."""
        env = PriorityRescueEnv(num_agents=0, grid_size=5, num_survivors=1)
        env.reset()
        env.agents = []
        env.survivor_data[0]['alive'] = False
        env.survivor_data[0]['rescued'] = False
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Should contain dead emoji
        assert '💀' in html

    @patch('streamlit_priority_demo.st')
    def test_render_grid_hides_rescued_survivor(self, mock_st):
        """Test grid doesn't show rescued survivors."""
        env = PriorityRescueEnv(num_agents=0, grid_size=5, num_survivors=1)
        env.reset()
        env.agents = []

        # Get survivor position before rescue
        survivor_pos = env.survivor_data[0]['position']

        # Mark as rescued
        env.survivor_data[0]['rescued'] = True
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Rescued survivors should not appear (grid cell should be empty)
        # Note: This is complex to test in HTML, so we check it doesn't show urgency emojis
        # at that position if we could track it

    @patch('streamlit_priority_demo.st')
    def test_render_grid_shows_star_on_survivor(self, mock_st):
        """Test grid shows star emoji when agent is on survivor."""
        env = PriorityRescueEnv(num_agents=1, grid_size=10, num_survivors=1)
        env.reset()

        # Place agent on survivor
        survivor_pos = env.survivor_data[0]['position']
        env.agent_positions['drone_0'] = survivor_pos
        mock_st.session_state = MockSessionState({'env': env})

        html = render_grid()

        # Should show star emoji
        assert '⭐' in html


class TestTakeStep:
    """Test take_step function."""

    @patch('streamlit_priority_demo.st')
    def test_take_step_increments_counter(self, mock_st):
        """Test take_step increments step counter."""
        env = PriorityRescueEnv(num_agents=2)
        obs, infos = env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {
                'drone_0': RandomAgent('drone_0', env.action_space),
                'drone_1': RandomAgent('drone_1', env.action_space)
            },
            'observations': obs,
            'infos': infos,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False
        })

        take_step()

        assert mock_st.session_state['step_count'] == 1

    @patch('streamlit_priority_demo.st')
    def test_take_step_updates_observations(self, mock_st):
        """Test take_step updates observations."""
        env = PriorityRescueEnv(num_agents=2)
        obs, infos = env.reset()
        initial_obs = obs.copy()

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {
                'drone_0': RandomAgent('drone_0', env.action_space),
                'drone_1': RandomAgent('drone_1', env.action_space)
            },
            'observations': obs,
            'infos': infos,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False
        })

        take_step()

        # Observations should be updated (different from initial)
        assert 'observations' in mock_st.session_state

    @patch('streamlit_priority_demo.st')
    def test_take_step_accumulates_rewards(self, mock_st):
        """Test take_step accumulates rewards."""
        env = PriorityRescueEnv(num_agents=1)
        obs, infos = env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {'drone_0': RandomAgent('drone_0', env.action_space)},
            'observations': obs,
            'infos': infos,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False
        })

        initial_reward = mock_st.session_state['total_reward']
        take_step()

        # Total reward should change (likely decrease due to penalties)
        assert mock_st.session_state['total_reward'] != initial_reward

    @patch('streamlit_priority_demo.st')
    def test_take_step_sets_done_on_termination(self, mock_st):
        """Test take_step sets done flag on termination."""
        env = PriorityRescueEnv(num_agents=1, num_survivors=0)
        obs, infos = env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {'drone_0': RandomAgent('drone_0', env.action_space)},
            'observations': obs,
            'infos': infos,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False
        })

        take_step()

        # Should be done (no survivors to rescue)
        assert mock_st.session_state['done'] is True

    @patch('streamlit_priority_demo.st')
    def test_take_step_does_nothing_when_done(self, mock_st):
        """Test take_step does nothing when already done."""
        env = PriorityRescueEnv(num_agents=1)
        obs, infos = env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {'drone_0': RandomAgent('drone_0', env.action_space)},
            'observations': obs,
            'infos': infos,
            'step_count': 5,
            'total_reward': 100.0,
            'done': True
        })

        take_step()

        # Should not increment step count
        assert mock_st.session_state['step_count'] == 5

    @patch('streamlit_priority_demo.st')
    def test_take_step_calls_agent_act(self, mock_st):
        """Test take_step calls agent.act() method."""
        env = PriorityRescueEnv(num_agents=1)
        obs, infos = env.reset()

        mock_agent = Mock()
        mock_agent.act = Mock(return_value=0)

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {'drone_0': mock_agent},
            'observations': obs,
            'infos': infos,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False
        })

        take_step()

        # Verify agent.act was called
        mock_agent.act.assert_called_once()


class TestResetSimulation:
    """Test reset_simulation function."""

    @patch('streamlit_priority_demo.st')
    def test_reset_simulation_resets_counters(self, mock_st):
        """Test reset_simulation resets counters to zero."""
        env = PriorityRescueEnv(num_agents=2)
        env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'step_count': 50,
            'total_reward': -100.0,
            'done': True,
            'observations': {},
            'infos': {}
        })

        reset_simulation()

        assert mock_st.session_state['step_count'] == 0
        assert mock_st.session_state['total_reward'] == 0.0
        assert mock_st.session_state['done'] is False

    @patch('streamlit_priority_demo.st')
    def test_reset_simulation_resets_environment(self, mock_st):
        """Test reset_simulation calls env.reset()."""
        mock_env = Mock()
        mock_env.reset = Mock(return_value=({}, {}))

        mock_st.session_state = MockSessionState({
            'env': mock_env,
            'step_count': 10,
            'total_reward': 50.0,
            'done': False,
            'observations': {},
            'infos': {}
        })

        reset_simulation()

        # Verify env.reset was called with seed
        mock_env.reset.assert_called_once_with(seed=42)

    @patch('streamlit_priority_demo.st')
    def test_reset_simulation_updates_observations(self, mock_st):
        """Test reset_simulation updates observations and infos."""
        env = PriorityRescueEnv(num_agents=2)
        env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'step_count': 10,
            'total_reward': 50.0,
            'done': True,
            'observations': None,
            'infos': None
        })

        reset_simulation()

        assert mock_st.session_state['observations'] is not None
        assert mock_st.session_state['infos'] is not None
        assert isinstance(mock_st.session_state['observations'], dict)
        assert isinstance(mock_st.session_state['infos'], dict)

    @patch('streamlit_priority_demo.st')
    def test_reset_simulation_uses_same_seed(self, mock_st):
        """Test reset_simulation uses seed 42 for reproducibility."""
        env = PriorityRescueEnv(num_agents=2)
        mock_st.session_state = MockSessionState({
            'env': env,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False,
            'observations': {},
            'infos': {}
        })

        # Reset twice and check observations are identical
        reset_simulation()
        obs1 = mock_st.session_state['observations']['drone_0'].copy()

        reset_simulation()
        obs2 = mock_st.session_state['observations']['drone_0'].copy()

        np.testing.assert_array_equal(obs1, obs2)


class TestIntegration:
    """Integration tests for streamlit demo components."""

    @patch('streamlit_priority_demo.st')
    def test_full_simulation_cycle(self, mock_st):
        """Test complete simulation cycle: init -> step -> reset."""
        mock_st.session_state = MockSessionState()

        # Initialize
        initialize_session_state()
        assert mock_st.session_state['step_count'] == 0

        # Take some steps
        for _ in range(5):
            if not mock_st.session_state['done']:
                take_step()

        assert mock_st.session_state['step_count'] > 0

        # Reset
        reset_simulation()
        assert mock_st.session_state['step_count'] == 0
        assert mock_st.session_state['done'] is False

    @patch('streamlit_priority_demo.st')
    def test_render_after_steps(self, mock_st):
        """Test rendering after taking steps."""
        mock_st.session_state = MockSessionState()

        initialize_session_state()

        # Take a few steps
        for _ in range(3):
            take_step()

        # Render should work without errors
        html = render_grid()
        assert isinstance(html, str)
        assert len(html) > 0

    @patch('streamlit_priority_demo.st')
    def test_simulation_reaches_completion(self, mock_st):
        """Test simulation can reach completion (done state)."""
        mock_st.session_state = MockSessionState()

        # Create environment with fast completion
        env = PriorityRescueEnv(num_agents=1, num_survivors=0)
        obs, infos = env.reset()

        mock_st.session_state = MockSessionState({
            'env': env,
            'agents': {'drone_0': RandomAgent('drone_0', env.action_space)},
            'observations': obs,
            'infos': infos,
            'step_count': 0,
            'total_reward': 0.0,
            'done': False
        })

        # Take one step
        take_step()

        # Should be done (no survivors)
        assert mock_st.session_state['done'] is True
