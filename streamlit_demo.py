"""
Streamlit visualizer for Project Argus

Interactive web-based visualization of the rescue mission.
"""

import streamlit as st
import numpy as np
from src.random_agent import RandomAgent
from env.weather_rescue import SimpleRescueEnv


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'env' not in st.session_state:
        st.session_state.env = SimpleRescueEnv(num_agents=3, grid_size=8)
        st.session_state.agents = {}
        for agent_id in st.session_state.env.possible_agents:
            st.session_state.agents[agent_id] = RandomAgent(
                agent_id, st.session_state.env.action_space
            )
        st.session_state.observations, st.session_state.infos = st.session_state.env.reset()
        st.session_state.step_count = 0
        st.session_state.total_reward = 0
        st.session_state.done = False


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
        
        # Display number of survivors
        st.metric("Survivors", len(st.session_state.env.survivor_positions))
        
        st.divider()
        
        # Drone positions
        st.subheader("Drone Positions")
        for agent_id in st.session_state.env.agents:
            pos = st.session_state.env.agent_positions[agent_id]
            st.text(f"{agent_id}: ({pos[0]}, {pos[1]})")
        
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
