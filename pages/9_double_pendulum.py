import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.physics_core import simulate_double_pendulum
from modules.diagrams import get_double_pendulum_schema

st.set_page_config(page_title="Double Pendulum", layout="wide")

st.title("🎢 Double Pendulum Dynamics")
st.markdown("""
This simulation demonstrates chaotic motion derived using the **Lagrangian Mechanics** toolkit.
Small changes in initial conditions lead to vastly different trajectories.
""")

# --- Sidebar Controls ---
st.sidebar.header("System Parameters")
m1 = st.sidebar.slider("Mass 1 (kg)", 0.1, 10.0, 1.0)
m2 = st.sidebar.slider("Mass 2 (kg)", 0.1, 10.0, 1.0)
L1 = st.sidebar.slider("Length 1 (m)", 0.1, 5.0, 1.0)
L2 = st.sidebar.slider("Length 2 (m)", 0.1, 5.0, 1.0)
g = st.sidebar.slider("Gravity (m/s²)", 1.0, 20.0, 9.81)

st.sidebar.header("Initial Conditions")
theta1_deg = st.sidebar.slider("Angle 1 (deg)", -180, 180, 90)
theta2_deg = st.sidebar.slider("Angle 2 (deg)", -180, 180, 0)
omega1 = st.sidebar.number_input("Velocity 1 (rad/s)", value=0.0)
omega2 = st.sidebar.number_input("Velocity 2 (rad/s)", value=0.0)

# Convert to radians
y0 = [np.radians(theta1_deg), omega1, np.radians(theta2_deg), omega2]

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Simulation")
    
    # Time setup
    duration = st.slider("Simulation Duration (s)", 1, 20, 10)
    t = np.linspace(0, duration, 1000)
    
    # Run Physics
    sol = simulate_double_pendulum(t, y0, g, m1, m2, L1, L2)
    theta1 = sol[:, 0]
    theta2 = sol[:, 2]

    # Calculate Cartesian coordinates for plotting
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # Animation Slider
    time_idx = st.slider("Time Step", 0, len(t)-1, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x2, y2, alpha=0.3, color='gray', label="Trace") # Trace path
    ax.plot([0, x1[time_idx], x2[time_idx]], [0, y1[time_idx], y2[time_idx]], 'o-', lw=2)
    ax.set_xlim(- (L1 + L2), (L1 + L2))
    ax.set_ylim(- (L1 + L2), (L1 + L2))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("System Diagram")
    # Render Graphviz
    st.graphviz_chart(get_double_pendulum_schema(), width='stretch')
    
    st.subheader("State Data")
    st.write(f"**Theta 1:** {np.degrees(theta1[time_idx]):.2f}°")
    st.write(f"**Theta 2:** {np.degrees(theta2[time_idx]):.2f}°")
    st.info("Note: This system is non-integrable and exhibits chaotic behavior.")
