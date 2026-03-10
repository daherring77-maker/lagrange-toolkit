import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.lagrange_points import LagrangePointSolver
from modules.diagrams import get_lagrange_multiplier_schema, get_lagrange_points_diagram

st.set_page_config(page_title="Lagrange Points", layout="centered")

st.title("🌌 Lagrange Points & Constrained Optimization")
st.markdown("""
This module bridges **Lagrangian Mechanics** and **State-Space Analysis**.
We use **Lagrange Multipliers** (constrained optimization) to find equilibrium points 
in the Circular Restricted Three-Body Problem (CR3BP).
""")

# Replace the sidebar controls section with this enhanced version:

# --- Sidebar Controls ---
st.sidebar.header("🌍 System Presets")
preset = st.sidebar.selectbox(
    "Choose a system:",
    list(LagrangePointSolver.PRESETS.keys()),
    index=1  # Default to Earth-Sun
)

if preset in LagrangePointSolver.PRESETS:
    params = LagrangePointSolver.PRESETS[preset]
    st.sidebar.info(f"**Mass Ratio μ:** {params['mu']:.2e}")
    st.sidebar.info(f"**Units:** Normalized (M=1, d=1, ω=1)")

# Advanced toggle
show_advanced = st.sidebar.checkbox("🔧 Advanced: Custom Parameters", value=False)

if show_advanced:
    m1 = st.sidebar.slider("Primary Mass (M₁)", 0.001, 1.0, params['m1'], format="%.4f")
    m2 = st.sidebar.slider("Secondary Mass (M₂)", 0.001, 1.0, params['m2'], format="%.4f")
    omega = st.sidebar.slider("Angular Velocity (ω)", 0.1, 3.0, params['omega'])
else:
    m1, m2, omega = params['m1'], params['m2'], params['omega']

# Initialize Solver
solver = LagrangePointSolver(m1=m1, m2=m2, omega=omega)

# Visualization zoom for small mu systems
if solver.mu < 0.01:
    st.sidebar.header("🔍 View Options")
    zoom = st.sidebar.radio(
        "Zoom level:",
        ["Full System", "Zoom: Earth-Sun Region", "Zoom: L1/L2 Region"],
        index=1 if preset == "Earth-Sun (μ=3e-6)" else 0
    )
else:
    zoom = "Full System"


# --- Main Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📐 Theory", "🔍 Optimization", "🗺️ Visualization", "🚀 State Space"])

with tab1:
    st.header("1. Lagrange Multipliers Theory")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("The Mathematical Framework")
        st.markdown("Problem: Minimize f(x,y) subject to g(x,y) = 0")
        st.markdown("Method: Introduce Lagrange multiplier λ")
        
        st.latex(r"\mathcal{L}(x, y, \lambda) = f(x,y) - \lambda \cdot g(x,y)")
        
        st.markdown("Solution: Set all partial derivatives to zero")
                
        st.markdown("""
         **Solution:** Solve ∇ℒ = 0           
        - ∂ℒ/∂x = 0
        - ∂ℒ/∂y = 0  
        - ∂ℒ/∂λ = 0
        """)
                
        
        st.subheader("Connection to Physics")
        st.markdown("""
        In the **Restricted 3-Body Problem**:
        - **Objective:** Minimize effective potential $\\Phi_{eff}$
        - **Constraint:** Force balance (∇Φ = 0)
        - **Result:** 5 equilibrium points (L1-L5)
        """)
    
    with col2:
        st.subheader("Optimization Flow")
        st.graphviz_chart(get_lagrange_multiplier_schema(), width='stretch')
    
    st.divider()
    
    st.header("2. The Five Lagrange Points")
    st.graphviz_chart(get_lagrange_points_diagram(), width='stretch')
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Collinear Points (L1, L2, L3)**\n\nUnstable equilibrium. Require station-keeping fuel.")
    with col2:
        st.success("**Triangular Points (L4, L5)**\n\nStable equilibrium (for certain mass ratios). Natural traps for asteroids.")

with tab2:
    st.header("Scipy Optimization Results")
    st.markdown("Finding stationary points using `scipy.optimize.minimize` (BFGS method).")
    
    # Run solver
    with st.spinner("Computing Lagrange points..."):
        points = solver.find_lagrange_points()
    
    # Display results table
    st.subheader("Point Coordinates & Stability")
    table_data = []
    for name, data in points.items():
        x, y = data['position']
        status = "✅ Stable" if data['stable'] else "❌ Unstable"
        table_data.append({
            "Point": name,
            "X": f"{x:.4f}",
            "Y": f"{y:.4f}",
            "Potential": f"{data['potential']:.4f}",
            "Stability": status
        })
    
    st.table(table_data)
    
    st.subheader("Optimization Convergence")
    for name, data in points.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(1.0 if data['converged'] else 0.5, text=f"{name} Convergence")
        with col2:
            st.metric("Status", "Success" if data['converged'] else "Failed")

# In Tab 3, replace the plotting section with adaptive bounds:

with tab3:
    st.header("Effective Potential Visualization")
    
    # Adaptive plotting bounds based on mu and zoom selection
    if solver.mu < 0.01 and zoom != "Full System":
        if zoom == "Zoom: Earth-Sun Region":
            x_bounds = [-0.05, 0.05]  # Focus on Earth's neighborhood
            y_bounds = [-0.03, 0.03]
            st.info("🔍 Zoomed view: Earth's orbital neighborhood")
        else:  # L1/L2 Region
            r_hill = (solver.mu/3)**(1/3)
            x_bounds = [1 - solver.mu - 2*r_hill, 1 - solver.mu + 2*r_hill]
            y_bounds = [-r_hill, r_hill]
            st.info(f"🔍 Zoomed view: L1/L2 region (Hill sphere ≈ {r_hill:.4f})")
    else:
        x_bounds = [-1.5, 1.5]
        y_bounds = [-1.0, 1.0]
    
    # Create grid with adaptive resolution
    x = np.linspace(x_bounds[0], x_bounds[1], 500)
    y = np.linspace(y_bounds[0], y_bounds[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Vectorized potential calculation for speed
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = solver.effective_potential([X[j, i], Y[j, i]])
    
    # Plot with adaptive styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use logarithmic contour spacing for small mu to show detail
    if solver.mu < 0.01:
        levels = np.linspace(np.min(Z), np.max(Z), 100)
    else:
        levels = 50
        
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Effective Potential')
    
    # Plot masses with size proportional to mass
    ax.plot(solver.r1, 0, 'o', color='#ffeb3b', 
            markersize=20*np.sqrt(solver.m1), 
            label=f'M₁ (Primary)', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(solver.r2, 0, 'o', color='#42a5f5', 
            markersize=20*np.sqrt(solver.m2),
            label=f'M₂ (Secondary)', markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot Lagrange points with enhanced visibility for small mu
    for name, data in points.items():
        px, py = data['position']
        color = '#2ecc71' if data['stable'] else '#e74c3c'
        marker_size = 12 if solver.mu > 0.01 else 8
        ax.plot(px, py, 'x', color=color, markersize=marker_size, 
                markeredgewidth=2.5, label=f'{name}')
        # Add label offset for crowded regions
        if solver.mu < 0.01 and name in ['L1', 'L2']:
            ax.annotate(name, (px, py+0.003), fontsize=8, color=color)
    
    ax.set_xlabel('X Position (normalized)')
    ax.set_ylabel('Y Position (normalized)')
    ax.set_title(f'Effective Potential - μ = {solver.mu:.2e}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    st.pyplot(fig)
    
    # Add physical scale info for Earth-Sun
    if preset == "Earth-Sun (μ=3e-6)":
        st.caption("""
        📏 **Physical Scale**: 1 normalized unit = 1 AU ≈ 149.6 million km  
        🛰️ **L1 Distance from Earth**: ≈ 1.5 million km (4× Moon distance)  
        🔭 **JWST orbits L2**, SOHO orbits L1
        """)
with tab4:
    st.header("State-Space Connection")
    st.markdown("""
    This is the bridge to **Hamiltonian Mechanics** and **Advanced System State**.
    
    At a Lagrange point, the system is in equilibrium:
    $$ \\text{State} = [x, y, \\dot{x}, \\dot{y}] = [x_L, y_L, 0, 0] $$
    
    To analyze stability, we **linearize** the equations of motion around this point:
    $$ \\dot{\\mathbf{x}} = A \\mathbf{x} $$
    Where $A$ is the Jacobian matrix evaluated at the equilibrium.
    """)
    
    st.subheader("Linear Stability Analysis")
    st.write("Eigenvalues of the Hessian determine local stability:")
    
    for name, data in points.items():
        with st.expander(f"{name} Eigenvalue Analysis"):
            eig = data['eigenvalues']
            st.write(f"**Eigenvalues:** {eig}")
            if data['stable']:
                st.success("Both positive → Local minimum → **Stable**")
            else:
                st.error("Mixed signs → Saddle point → **Unstable**")
    
    st.divider()
    
    st.subheader("Preview: Hamiltonian Formulation")
    st.info("""
    **Coming in Advanced Module:**
    
    We will convert this Lagrangian system to Hamiltonian form:
    $$ H = \\sum p_i \\dot{q}_i - L $$
    
    This gives us **phase space** trajectories and reveals:
    - Stable orbits around L4/L5 (tadpole & horseshoe orbits)
    - Unstable manifolds near L1/L2 (used for low-fuel transfers)
    - Conservation of Jacobi Integral
    """)
    
    st.write("This connects directly to your **System State** studies!")

# --- Footer ---
st.divider()
st.markdown("""
**Next Steps:** 
- Explore [Double Pendulum](/Double_Pendulum) for chaotic Lagrangian systems
- Study [Euler-Lagrange](/Euler_Lagrange) for derivation methods
- **Coming Soon:** Hamiltonian Phase Space Analysis
""")