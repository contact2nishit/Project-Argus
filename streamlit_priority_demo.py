"""
Streamlit visualizer for Priority Rescue Environment

Interactive web-based visualization of priority-based rescue with time decay.
"""

import streamlit as st

from time import sleep

from src.random_agent import RandomAgent
from env.priority_rescue import PriorityRescueEnv


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'env' not in st.session_state:
        st.session_state.env = PriorityRescueEnv(
            num_agents=3,
            grid_size=15,
            num_survivors=6,
            max_steps=200
        )
        st.session_state.agents = {}
        for agent_id in st.session_state.env.possible_agents:
            st.session_state.agents[agent_id] = RandomAgent(
                agent_id, st.session_state.env.action_space
            )
        st.session_state.observations, st.session_state.infos = st.session_state.env.reset(seed=42)
        st.session_state.step_count = 0
        st.session_state.total_reward = 0
        st.session_state.done = False


def get_urgency_emoji(urgency_level):
    """Get emoji for urgency level."""
    urgency_map = {
        0: '🟢',  # LOW - green
        1: '🟡',  # MEDIUM - yellow
        2: '🟠',  # HIGH - orange
        3: '🔴',  # CRITICAL - red
    }
    return urgency_map.get(urgency_level, '⚪')


def get_urgency_name(urgency_level):
    """Get name for urgency level."""
    names = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    return names.get(urgency_level, "UNKNOWN")


def render_grid():
    """Render the grid in Streamlit using HTML/CSS."""
    env = st.session_state.env
    grid_size = env.grid_size

    # Create empty grid
    grid = [['⬜' for _ in range(grid_size)] for _ in range(grid_size)]

    # Place survivors with urgency-based colors
    for survivor in env.survivor_data:
        if survivor['rescued']:
            continue  # Don't show rescued survivors

        row, col = survivor['position']
        if not survivor['alive']:
            grid[row][col] = '💀'  # Dead survivor
        else:
            # Show urgency emoji
            emoji = get_urgency_emoji(survivor['urgency'])
            grid[row][col] = emoji

    # Place agents (overwrite if on same position as survivor)
    agent_emojis = ['🚁', '🛸', '✈️']
    for idx, agent_id in enumerate(env.agents):
        pos = env.agent_positions[agent_id]
        row, col = pos

        # Check if agent is on survivor location
        on_survivor = False
        for survivor in env.survivor_data:
            if survivor['position'] == [row, col] and survivor['alive'] and not survivor['rescued']:
                on_survivor = True
                break

        if on_survivor:
            grid[row][col] = '⭐'  # Agent on survivor
        else:
            grid[row][col] = agent_emojis[idx % len(agent_emojis)]

    # Build HTML table with smaller cells for 15x15 grid
    html = '<div style="display: flex; justify-content: center; margin: 20px 0;">'
    html += '<table style="border-collapse: collapse; font-size: 20px;">'

    # Header row with column numbers
    html += '<tr><td style="padding: 2px; text-align: center; font-size: 12px;"></td>'
    for col in range(grid_size):
        html += f'<td style="padding: 2px; text-align: center; font-size: 10px; font-weight: bold;">{col}</td>'
    html += '</tr>'

    # Grid rows
    for row_idx, row in enumerate(grid):
        html += '<tr>'
        # Row number
        html += f'<td style="padding: 2px; text-align: center; font-size: 10px; font-weight: bold;">{row_idx}</td>'
        # Grid cells
        for cell in row:
            html += f'<td style="padding: 2px; text-align: center; border: 1px solid #ddd; width: 35px; height: 35px;">{cell}</td>'
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
    st.session_state.observations, st.session_state.infos = st.session_state.env.reset(seed=42)
    st.session_state.step_count = 0
    st.session_state.total_reward = 0
    st.session_state.done = False


def main():
    st.set_page_config(
        page_title="Priority Rescue - Project Argus",
        page_icon="🚁",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Title and description
    st.title("🚁 Priority Rescue Mission")
    st.markdown("**Multi-Agent RL with Time-Critical Triage & Health Decay**")
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
        - 🚁 🛸 ✈️ Drones
        - 🟢 Low Urgency | 🟡 Medium | 🟠 High | 🔴 Critical
        - ⭐ Drone on survivor!
        - 💀 Dead survivor
        - ⬜ Empty space
        """)

    with col2:
        st.subheader("Mission Status")

        # Display metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Step", f"{st.session_state.step_count}/{st.session_state.env.max_steps}")
        with col_m2:
            st.metric("Total Reward", f"{st.session_state.total_reward:.1f}")

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.metric("Rescued", f"{st.session_state.env.rescued_count}/{st.session_state.env.num_survivors}")
        with col_m4:
            st.metric("Deaths", st.session_state.env.death_count)

        st.divider()

        # Survivor status
        st.subheader("Survivor Status")

        # Sort survivors by urgency (critical first)
        survivors = sorted(
            st.session_state.env.survivor_data,
            key=lambda s: (-s['urgency'], s['health'])
        )

        for i, survivor in enumerate(survivors):
            if survivor['rescued']:
                st.success(f"✅ Survivor {i}: RESCUED (+{survivor['rescue_reward']})")
            elif not survivor['alive']:
                st.error(f"💀 Survivor {i}: DEAD")
            else:
                urgency_name = get_urgency_name(survivor['urgency'])
                emoji = get_urgency_emoji(survivor['urgency'])
                health_pct = (survivor['health'] / survivor['max_health']) * 100

                # Color code based on health
                if health_pct < 20:
                    color = "🔴"
                elif health_pct < 50:
                    color = "🟠"
                else:
                    color = ""

                st.text(f"{emoji} {urgency_name:8s} | HP: {survivor['health']:5.1f}/{survivor['max_health']:.0f} ({health_pct:.0f}%) {color}")

        st.divider()

        # Drone positions
        st.subheader("Drone Positions")
        for agent_id in st.session_state.env.agents:
            pos = st.session_state.env.agent_positions[agent_id]
            st.text(f"{agent_id}: ({pos[0]:2d}, {pos[1]:2d})")

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

        # Multi-step button
        if st.button("⏩ Next 5 Steps", disabled=st.session_state.done, use_container_width=True):
            for _ in range(5):
                if not st.session_state.done:
                    take_step()
            st.rerun()

        if st.session_state.done:
            success_rate = (st.session_state.env.rescued_count / st.session_state.env.num_survivors) * 100
            if success_rate >= 80:
                st.success(f"✅ Mission Success! {success_rate:.0f}% rescued")
            elif success_rate >= 50:
                st.warning(f"⚠️ Partial Success: {success_rate:.0f}% rescued")
            else:
                st.error(f"❌ Mission Failed: {success_rate:.0f}% rescued")

        # Auto-step option
        st.divider()
        auto_step = st.checkbox("Auto-step (0.5 sec delay)")
        if auto_step and not st.session_state.done:
            
            sleep(0.5)
            take_step()
            st.rerun()


if __name__ == '__main__':
    main()
