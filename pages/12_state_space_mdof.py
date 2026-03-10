import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.state_space import MDOFSystem

st.set_page_config(page_title="State-Space & MDOF", layout="wide")

st.title("📊 State-Space & Multi-DOF Systems")
st.markdown("""
Bridging **Lagrangian Mechanics** to **Modern Control Theory**.
We analyze a spring-mass-damper chain to understand eigenvalues, stability, and mode shapes.
""")

# --- Sidebar Controls ---
st.sidebar.header("System Configuration")
n_dof = st.sidebar.slider("Degrees of Freedom (N)", 1, 3, 2)

st.sidebar.header("Parameters (Per Mass)")
m = st.sidebar.slider("Mass (kg)", 0.5, 5.0, 1.0)
k = st.sidebar.slider("Stiffness (N/m)", 10, 500, 100)
c = st.sidebar.slider("Damping (N·s/m)", 0, 50, 5)

# Create arrays for N-DOF
masses = [m] * n_dof
stiffness = [k] * (n_dof + 1)  # +1 for ground spring
damping = [c] * (n_dof + 1)

# Initialize System
system = MDOFSystem(masses, stiffness, damping)

# --- Main Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["🔧 Physical System", "🧮 State Formulation", "📈 Eigenvalues", "🔗 Lagrangian Link"])

with tab1:
    st.header("1. Physical Intuition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("System Diagram")
        # Simple ASCII-style visualization using Graphviz
        dot_code = f"""
        digraph MDOF {{
            rankdir=LR;
            node [shape=box, style=filled, fillcolor=lightblue];
            Ground [shape=plaintext, label="Ground"];
            {"".join([f'M{i+1} [label="m{i+1}"];' for i in range(n_dof)])}
            Ground -> M1 [label="k1, c1"];
            {"".join([f'M{i+1} -> M{i+2} [label="k{i+2}, c{i+2}"];' for i in range(n_dof-1)])}
        }}
        """
        st.graphviz_chart(dot_code, width='stretch')
        
        st.info(f"**Degrees of Freedom:** {n_dof}\n**Total States:** {2*n_dof} (Position + Velocity)")
    
    with col2:
        st.subheader("Mode Shapes")
        st.markdown("How the system naturally vibrates.")
        
        mode_shapes = system.get_mode_shapes()
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot first 2 mode shapes
        for i in range(min(2, len(mode_shapes[0]))):
            # Real part of eigenvector
            shape = np.real(mode_shapes[:, i])
            # Normalize
            shape = shape / np.max(np.abs(shape))
            ax.plot(range(1, n_dof+1), shape, 'o-', label=f'Mode {i+1}')
        
        ax.set_xlabel('Mass Index')
        ax.set_ylabel('Relative Displacement')
        ax.set_title('Normalized Mode Shapes')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(range(1, n_dof+1))
        
        st.pyplot(fig)

with tab2:
    st.header("2. State-Space Formulation")
    
    st.markdown("""
    **The Challenge:** Newton's laws give us 2nd-order ODEs:
    $$ M\\ddot{x} + C\\dot{x} + Kx = 0 $$
    
    **The Solution:** Convert to 1st-order system by defining state vector $\\mathbf{z}$:
    """)
    
    st.latex(r"""
        \mathbf{z} = \begin{bmatrix} x \\ \dot{x} \end{bmatrix}, \quad 
        \dot{\mathbf{z}} = \mathbf{A}\mathbf{z}
    """)
    
    st.markdown("Where the **State Matrix A** is:")
    
    # Show A matrix structure
    st.latex(r"""
        \mathbf{A} = \begin{bmatrix} 
            0 & I \\ 
            -M^{-1}K & -M^{-1}C 
        \end{bmatrix}
    """)
    
    st.subheader("Your System's A Matrix")
    st.write(f"Shape: {system.A.shape}")
    st.dataframe(np.round(system.A, 3), width='stretch')
    
    st.success("""
    **Why do this?** 
    1. Computers solve 1st-order systems efficiently
    2. Eigenvalue analysis becomes straightforward
    3. Extends to control theory (controllability, observability)
    """)

with tab3:
    st.header("3. Eigenvalue Analysis")
    
    st.markdown("""
    Eigenvalues ($\\lambda$) contain the **physical meaning** of the system dynamics.
    $$ \\lambda = \\sigma \\pm i\\omega $$
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Eigenvalue Properties")
        
        table_data = []
        for i, ev in enumerate(system.eigenvalues[:n_dof]):  # Show unique pairs
            freq = system.get_natural_frequencies()[i]
            zeta = system.get_damping_ratios()[i]
            stability = "✅ Stable" if np.real(ev) < 0 else "❌ Unstable"
            
            table_data.append({
                "Mode": i+1,
                "Eigenvalue": f"{ev:.3f}",
                "Freq (Hz)": f"{freq:.3f}",
                "Damping ζ": f"{zeta:.3f}",
                "Stability": stability
            })
        
        st.table(table_data)
    
    with col2:
        st.subheader("Eigenvalue Plot (S-Plane)")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot eigenvalues
        ax.plot(np.real(system.eigenvalues), np.imag(system.eigenvalues), 'x', 
                markersize=15, markeredgewidth=2, label='Eigenvalues')
        ax.plot(np.real(system.eigenvalues), -np.imag(system.eigenvalues), 'x', 
                markersize=15, markeredgewidth=2)
        
        # Stability boundary
        ax.axvline(0, color='red', linestyle='--', label='Stability Boundary')
        
        ax.set_xlabel('Real (σ) - Decay Rate')
        ax.set_ylabel('Imaginary (ω) - Frequency')
        ax.set_title('Eigenvalues in Complex Plane')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Parameter Sensitivity")
    st.markdown("See how eigenvalues move as you change damping:")
    
    # Create a sweep plot
    c_values = np.linspace(0, 50, 50)
    freq_track = []
    damp_track = []
    
    for cv in c_values:
        temp_sys = MDOFSystem(masses, stiffness, [cv]*(n_dof+1))
        freq_track.append(temp_sys.get_natural_frequencies()[0])
        damp_track.append(temp_sys.get_damping_ratios()[0])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(c_values, freq_track)
    ax[0].set_xlabel('Damping Coefficient')
    ax[0].set_ylabel('Natural Frequency (Hz)')
    ax[0].set_title('Frequency vs Damping')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(c_values, damp_track)
    ax[1].set_xlabel('Damping Coefficient')
    ax[1].set_ylabel('Damping Ratio (ζ)')
    ax[1].set_title('Damping Ratio vs Damping')
    ax[1].grid(True, alpha=0.3)
    ax[1].axhline(1.0, color='red', linestyle='--', label='Critical Damping')
    ax[1].legend()
    
    st.pyplot(fig)

with tab4:
    st.header("4. Connection to Lagrangian & Hamiltonian")
    
    st.markdown("""
    You've already learned the **Lagrangian** approach ($L = T - V$). 
    How does State-Space relate?
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("From Lagrangian to State-Space")
        st.markdown("""
        1. **Lagrangian:** Derive EOM using energy
           $$ \\frac{d}{dt}\\left(\\frac{\\partial L}{\\partial \\dot{q}}\\right) - \\frac{\\partial L}{\\partial q} = 0 $$
        
        2. **Linearize:** Assume small motions → $M\\ddot{x} + Kx = 0$
        
        3. **State-Space:** Convert to 1st-order form
           $$ \\dot{\\mathbf{z}} = \\mathbf{A}\\mathbf{z} $$
        """)
    
    with col2:
        st.subheader("Preview: Hamiltonian")
        st.info("""
        **Coming in Advanced Module:**
        
        The **Hamiltonian** $H = T + V$ leads to **Phase Space**:
        $$ \\dot{q} = \\frac{\\partial H}{\\partial p}, \\quad \\dot{p} = -\\frac{\\partial H}{\\partial q} $$
        
        This gives us:
        - Conservation of energy trajectories
        - Symplectic structure
        - Canonical transformations
        """)
        
        st.graphviz_chart("""
        digraph Connection {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor=white];
            
            Lagrangian [label="Lagrangian\n(Energy Method)", fillcolor="#e3f2fd"];
            EOM [label="Equations of\nMotion (2nd order)", fillcolor="#fff3e0"];
            StateSpace [label="State-Space\n(1st order)", fillcolor="#e8f5e9"];
            Hamiltonian [label="Hamiltonian\n(Phase Space)", fillcolor="#fce4ec"];
            
            Lagrangian -> EOM;
            EOM -> StateSpace;
            StateSpace -> Hamiltonian [style=dashed, label="Next"];
        }
        """, width='stretch')
    
    st.divider()
    
    st.subheader("Why This Matters")
    st.markdown("""
    - **Control Engineering:** Design controllers using pole placement (eigenvalues)
    - **Structural Analysis:** Predict building sway, bridge vibrations
    - **Robotics:** Analyze manipulator dynamics
    - **Aerospace:** Flutter analysis, satellite attitude control
    
    **You now have the tools to analyze ANY linear dynamic system!**
    """)

# --- Footer ---
st.divider()
st.markdown("""
**Previous:** [Lagrange Points](/Lagrange_Points) | **Next:** Hamiltonian Phase Space (Coming Soon)
""")