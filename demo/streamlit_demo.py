"""
Streamlit visualizer for Project Argus

Interactive web-based visualization of the rescue mission.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.random_agent import RandomAgent
from src.heuristic_agent import HeuristicAgent
from env.simple_rescue import SimpleRescueEnv


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'env' not in st.session_state:
        st.session_state.env = SimpleRescueEnv(num_agents=3, grid_size=8)
        st.session_state.agent_type = "Random"
        st.session_state.agents = {}
        st.session_state.observations = None
        st.session_state.infos = None
        st.session_state.step_count = 0
        st.session_state.total_reward = 0
        st.session_state.done = False


def create_agents(agent_type):
    """Create agents based on selected type."""
    agents = {}
    AgentClass = RandomAgent if agent_type == "Random" else HeuristicAgent
    
    for agent_id in st.session_state.env.possible_agents:
        agents[agent_id] = AgentClass(
            agent_id, st.session_state.env.action_space
        )
    return agents


def render_grid():
    """Render the grid in Streamlit using HTML/CSS."""
    env = st.session_state.env
    grid_size = env.grid_size
    
    # Create empty grid
    grid = [['⬜' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Place survivors
    for survivor_pos in env.survivor_positions:
        row, col = survivor_pos
        grid[row][col] = '🔴'
    
    # Place agents (overwrite if on same position as survivor)
    agent_emojis = ['🔵', '🟢', '🟡']
    for idx, agent_id in enumerate(env.agents):
        pos = env.agent_positions[agent_id]
        row, col = pos
        # Check if agent is on survivor
        if [row, col] in env.survivor_positions:
            grid[row][col] = '⭐'
        else:
            grid[row][col] = agent_emojis[idx % len(agent_emojis)]
    
    # Build HTML table
    html = '<div style="display: flex; justify-content: center; margin: 20px 0;">'
    html += '<table style="border-collapse: collapse; font-size: 32px;">'
    
    # Header row with column numbers
    html += '<tr><td style="padding: 5px; text-align: center; font-size: 16px;"></td>'
    for col in range(grid_size):
        html += f'<td style="padding: 5px; text-align: center; font-size: 16px; font-weight: bold;">{col}</td>'
    html += '</tr>'
    
    # Grid rows
    for row_idx, row in enumerate(grid):
        html += '<tr>'
        # Row number
        html += f'<td style="padding: 5px; text-align: center; font-size: 16px; font-weight: bold;">{row_idx}</td>'
        # Grid cells
        for cell in row:
            html += f'<td style="padding: 5px; text-align: center; border: 1px solid #ddd; width: 50px; height: 50px;">{cell}</td>'
        html += '</tr>'
    
    html += '</table></div>'
    
    return html


def take_step():
    """Execute one step in the simulation."""
    if st.session_state.done:
        return
    
    env = st.session_state.env
    agents = st.session_state.agents
    observations = st.session_state.observations
    
    # Get actions from agents
    actions = {}
    for agent_id, agent in agents.items():
        if agent_id in observations:
            actions[agent_id] = agent.act(observations[agent_id])
    
    # Step environment
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Update session state
    st.session_state.observations = observations
    st.session_state.infos = infos
    st.session_state.step_count += 1
    st.session_state.total_reward += sum(rewards.values())
    
    # Check if done
    if any(terminations.values()) or any(truncations.values()):
        st.session_state.done = True


def reset_simulation():
    """Reset the simulation to initial state."""
    # Recreate agents with current type
    st.session_state.agents = create_agents(st.session_state.agent_type)
    st.session_state.observations, st.session_state.infos = st.session_state.env.reset()
    st.session_state.step_count = 0
    st.session_state.total_reward = 0
    st.session_state.done = False


def main():
    st.set_page_config(page_title="Project Argus - Drone Rescue", page_icon="🚁", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🚁 Project Argus - Drone Rescue Mission")
    st.markdown("**Multi-Agent Reinforcement Learning for Search & Rescue Operations**")
    st.divider()
    
    # Agent type selector in sidebar
    st.sidebar.header("🤖 Agent Configuration")
    agent_type = st.sidebar.radio(
        "Select Agent Type:",
        ["Random", "Heuristic"],
        index=0 if st.session_state.agent_type == "Random" else 1
    )
    
    # If agent type changed, update and reset
    if agent_type != st.session_state.agent_type:
        st.session_state.agent_type = agent_type
        reset_simulation()
    
    # Initialize agents if needed
    if st.session_state.observations is None:
        reset_simulation()
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    **Agent Types:**
    
    **Random Agent:**
    - Ignores observations
    - Takes random actions
    - Baseline for comparison
    
    **Heuristic Agent:**
    - Uses observations
    - Deterministic decisions
    - Compares obs values:
      - `obs[0] vs obs[1]` → vertical
      - `obs[2] vs obs[3]` → horizontal
    - Moves in strongest direction
    
    **Reward System:**
    - 🎉 **+10.0** for rescuing a survivor
    - 📍 **+0.1** per step closer to survivor
    - 📍 **-0.1** per step away from survivor
    - ⏱️ **-0.01** time penalty per step
    """)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Mission Grid")
        # Render grid
        grid_html = render_grid()
        st.markdown(grid_html, unsafe_allow_html=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🔵 🟢 🟡 Drones
        - 🔴 Survivors
        - ⭐ Drone on survivor!
        - ⬜ Empty space
        """)
    
    with col2:
        st.subheader("Mission Status")
        
        # Display step count
        st.metric("Step", st.session_state.step_count)
        
        # Display total reward
        st.metric("Total Reward", f"{st.session_state.total_reward:.2f}")
        
        # Display survivors info
        total_survivors = len(st.session_state.env.survivor_positions)
        rescued = len(st.session_state.env.rescued_survivors)
        st.metric("Survivors", f"{rescued}/{total_survivors} rescued")
        
        st.divider()
        
        # Show current agent type
        st.subheader("Current Agent")
        agent_emoji = "🎲" if st.session_state.agent_type == "Random" else "🧠"
        st.markdown(f"**{agent_emoji} {st.session_state.agent_type} Agent**")
        
        st.divider()
        
        # Drone positions and observations
        st.subheader("Drone Info")
        for idx, agent_id in enumerate(st.session_state.env.agents):
            pos = st.session_state.env.agent_positions[agent_id]
            
            # Check if this drone rescued anyone
            rescued_marker = ""
            if st.session_state.infos and agent_id in st.session_state.infos:
                if st.session_state.infos[agent_id].get('rescued', False):
                    rescued_marker = " 🎉"
            
            with st.expander(f"{agent_id}: ({pos[0]}, {pos[1]}){rescued_marker}"):
                if st.session_state.observations and agent_id in st.session_state.observations:
                    obs = st.session_state.observations[agent_id]
                    st.text(f"Observation:")
                    st.text(f"  [{obs[0]:.3f}, {obs[1]:.3f},")
                    st.text(f"   {obs[2]:.3f}, {obs[3]:.3f}]")
                    
                    if st.session_state.agent_type == "Heuristic":
                        vertical = obs[0] - obs[1]
                        horizontal = obs[2] - obs[3]
                        st.text(f"\nHeuristic signals:")
                        st.text(f"  Vertical: {vertical:+.3f}")
                        st.text(f"  Horizontal: {horizontal:+.3f}")
                    
                    # Show distance to nearest survivor
                    if st.session_state.infos and agent_id in st.session_state.infos:
                        dist = st.session_state.infos[agent_id].get('nearest_survivor_distance', 0)
                        st.text(f"\nDistance to survivor: {dist}")
        
        st.divider()
        
        # Control buttons
        st.subheader("Controls")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("▶️ Next Step", disabled=st.session_state.done, use_container_width=True):
                take_step()
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_simulation()
                st.rerun()
        
        if st.session_state.done:
            st.success("✅ Mission Complete!")
        
        # Auto-step option
        st.divider()
        auto_step = st.checkbox("Auto-step (1 sec delay)")
        if auto_step and not st.session_state.done:
            import time
            time.sleep(1)
            take_step()
            st.rerun()


if __name__ == '__main__':
    main()
