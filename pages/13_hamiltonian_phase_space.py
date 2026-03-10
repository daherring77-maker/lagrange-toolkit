# pages/5_🌀_Hamiltonian_Phase_Space.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx
import time

st.set_page_config(page_title="Hamiltonian Mechanics", layout="wide")

st.title("🌀 Hamiltonian Mechanics & Phase Space")
st.markdown("""
The transition from **Lagrangian** to **Hamiltonian** mechanics shifts our focus from 
**Forces** to **Energy**, and from **Configuration Space** to **Phase Space**.
""")

# --- Sidebar ---
st.sidebar.header("System Selection")
system_type = st.sidebar.selectbox(
    "Choose System:",
    ["Harmonic Oscillator", "Simple Pendulum", "Double Pendulum (Chaos)"]
)

st.sidebar.header("Simulation Parameters")
t_max = st.sidebar.slider("Simulation Time", 10, 100, 50)
n_trajectories = st.sidebar.slider("Number of Trajectories", 5, 50, 20)

# --- Physics Functions ---
def harmonic_oscillator(t, z, k=1.0, m=1.0):
    q, p = z
    dqdt = p / m
    dpdt = -k * q
    return [dqdt, dpdt]

def pendulum(t, z, g=9.81, L=1.0):
    theta, p = z
    dtheta = p / (L**2)  # Simplified momentum
    dpdt = -g * L * np.sin(theta)
    return [dtheta, dpdt]

def hamiltonian_oscillator(q, p, k=1.0, m=1.0):
    return (p**2 / (2*m)) + (0.5 * k * q**2)

def hamiltonian_pendulum(theta, p, g=9.81, L=1.0):
    return (p**2 / (2 * L**2)) - g * L * np.cos(theta)

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📐 Derivation", "🌀 Phase Space", "⚖️ Laplace vs. Eigen", "💻 Compute Stress Test"])

with tab1:
    st.header("1. From Lagrangian to Hamiltonian")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("The Legendre Transformation")
        st.latex(r"L(q, \dot{q}) = T - V")
        st.markdown("Define generalized momentum:")
        st.latex(r"p = \frac{\partial L}{\partial \dot{q}}")
        st.markdown("Transform to Hamiltonian:")
        st.latex(r"H(q, p) = p\dot{q} - L")
        
        st.info("💡 **Key Insight:** We swap velocity ($\\dot{q}$) for momentum ($p$) as the independent variable.")
    
    with col2:
        st.subheader("Conditions for H = E")
        st.markdown("""
        For Hamiltonian to equal Total Energy ($T+V$):
        1. ✅ Time-independent constraints
        2. ✅ Velocity-independent potential
        3. ✅ Quadratic kinetic energy
        """)
        st.graphviz_chart("""
        digraph Transform {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fillcolor=white];
            L [label="Lagrangian\n(q, q̇)", fillcolor="#e3f2fd"];
            Legendre [label="Legendre\nTransform", fillcolor="#fff3e0"];
            H [label="Hamiltonian\n(q, p)", fillcolor="#e8f5e9"];
            L -> Legendre -> H;
        }
        """, width='stretch')

with tab2:
    st.header("2. Phase Space Visualization")
    st.markdown("Phase space reveals global behavior (stability, chaos) that time-series hide.")
    
    # Select ODE
    if system_type == "Harmonic Oscillator":
        ode_func = harmonic_oscillator
        H_func = hamiltonian_oscillator
        q_label, p_label = "Position (q)", "Momentum (p)"
        q_range, p_range = [-3, 3], [-3, 3]
    else:
        ode_func = pendulum
        H_func = hamiltonian_pendulum
        q_label, p_label = "Angle (θ)", "Angular Momentum (p)"
        q_range, p_range = [-np.pi, np.pi], [-5, 5]
    
    # Generate Trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    for i in range(n_trajectories):
        q0 = np.random.uniform(q_range[0], q_range[1])
        p0 = np.random.uniform(p_range[0], p_range[1])
        z0 = [q0, p0]
        
        sol = solve_ivp(ode_func, [0, t_max], z0, t_eval=np.linspace(0, t_max, 1000))
        
        # Color by Energy
        energy = H_func(q0, p0)
        ax.plot(sol.y[0], sol.y[1], color=colors[i % len(colors)], alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel(q_label)
    ax.set_ylabel(p_label)
    ax.set_title(f'Phase Portrait: {system_type}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_range)
    ax.set_ylim(p_range)
    
    if system_type == "Simple Pendulum":
        ax.axhline(np.sqrt(2*9.81), color='red', linestyle='--', alpha=0.5, label='Separatrix')
        ax.axhline(-np.sqrt(2*9.81), color='red', linestyle='--', alpha=0.5)
        ax.legend()
    
    st.pyplot(fig)
    
    st.caption("🔴 **Separatrix:** The boundary between oscillation (closed orbits) and rotation (open trajectories).")

with tab3:
    st.header("3. Control (Laplace) vs. Dynamics (Eigenvalues)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Control Theory (Laplace)")
        st.markdown("""
        - **Focus:** Input-Output relationship
        - **Tool:** Transfer Function $G(s)$
        - **Analysis:** Step Response, Bode Plot
        - **Stability:** Poles in Left Half Plane
        - **Best For:** Designing controllers (PID, State Feedback)
        """)
        
        # Simple Step Response Plot
        t = np.linspace(0, 10, 100)
        zeta = 0.5
        wn = 1.0
        y = 1 - (np.exp(-zeta*wn*t) / np.sqrt(1-zeta**2)) * np.sin(wn*np.sqrt(1-zeta**2)*t + np.arccos(zeta))
        
        fig1, ax1 = plt.subplots()
        ax1.plot(t, y)
        ax1.set_title("Step Response (Laplace Domain)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Output")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
    
    with col2:
        st.subheader("🔬 Dynamics (Eigenvalues)")
        st.markdown("""
        - **Focus:** Internal System Behavior
        - **Tool:** State Matrix $A$ or Hamiltonian $H$
        - **Analysis:** Phase Portraits, Mode Shapes
        - **Stability:** Eigenvalue Real Parts < 0
        - **Best For:** Understanding natural modes, chaos, energy conservation
        """)
        
        # Simple Eigenvalue Plot
        fig2, ax2 = plt.subplots()
        ax2.plot([-1, -0.5, -2], [1, -1, 2], 'x', markersize=15)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_title("Eigenvalues (S-Plane / Phase Space)")
        ax2.set_xlabel("Real (Stability)")
        ax2.set_ylabel("Imaginary (Frequency)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    st.divider()
    st.info("""
    **Rule of Thumb:** 
    - Use **Laplace** when you need to *control* the system (make it follow a command).
    - Use **Hamiltonian/Eigen** when you need to *understand* the system (predict stability/chaos).
    """)

with tab4:
    st.header("4. Computational Stress Test (NetworkX)")
    st.markdown("""
    How does computation time scale with system complexity?
    We generate random spring-mass graphs and solve for eigenvalues.
    **Theory:** Eigenvalue decomposition scales as $O(N^3)$.
    """)
    
    max_nodes = st.slider("Max System Size (N)", 10, 2009, 1000)
    
    if st.button("🚀 Run Benchmark"):
        sizes = np.linspace(10, max_nodes, 10, dtype=int)
        times = []
        
        progress_bar = st.progress(0)
        
        for i, n in enumerate(sizes):
            start = time.time()
            
            # Generate random graph
            G = nx.random_geometric_graph(n, 0.2)
            
            # Assemble sparse stiffness matrix (simplified)
            # In reality, this would be a full FEA assembly
            K = nx.laplacian_matrix(G).toarray() + np.eye(n) * 0.1
            M = np.eye(n)
            
            # Solve eigenvalues
            np.linalg.eigvals(K)
            
            elapsed = time.time() - start
            times.append(elapsed)
            progress_bar.progress((i + 1) / len(sizes))
        
        # Plot Results
        fig, ax = plt.subplots()
        ax.loglog(sizes, times, 'o-', linewidth=2)
        ax.loglog(sizes, 1e-6 * sizes**3, 'r--', label='O(N³) Reference')
        ax.set_xlabel('System Size (N)')
        ax.set_ylabel('Computation Time (s)')
        ax.set_title('Computational Complexity: Eigenvalue Solver')
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        
        st.pyplot(fig)
        
        st.success(f"Benchmark Complete! Largest system ({max_nodes} DOF) solved in {times[-1]:.3f}s")
        st.caption("Note: Real FEA uses sparse solvers which are much faster than dense `np.linalg`.")

# --- Footer ---
st.divider()
st.markdown("**Previous:** [State-Space MDOF](/State_Space_MDOF) | **Next:** Advanced Chaos & Fractals (Coming Soon)")