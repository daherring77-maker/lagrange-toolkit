import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import sympy as sp

st.set_page_config(page_title="Lagrange: Rod in Bowl (Statics)", layout="centered")

st.title("🎯 Lagrangian Mechanics: Rod in a Bowl (Statics)")
st.markdown("""
### Introduction to Lagrange's Method

This example demonstrates **Lagrange's method for static equilibrium** using the principle of minimum potential energy. 

**Problem**: A uniform rod of length L and mass m rests inside a frictionless hemispherical bowl of radius R. Find the equilibrium angle θ.

**Key Insight**: For statics, we minimize potential energy V (since T=0, the Lagrangian L = -V).
""")

# Sidebar controls
st.sidebar.header("📐 System Parameters")

# Physical parameters with sliders
L = st.sidebar.slider("Rod Length (L)", 0.1, 2.0, 1.0, 0.1, 
                     help="Length of the rod")
R = st.sidebar.slider("Bowl Radius (R)", 0.5, 3.0, 1.5, 0.1,
                     help="Radius of the hemispherical bowl")
m = st.sidebar.slider("Rod Mass (m)", 0.1, 10.0, 2.0, 0.5,
                     help="Mass of the rod (kg)")
g = st.sidebar.slider("Gravity (g)", 1.0, 20.0, 9.81, 0.5,
                     help="Gravitational acceleration (m/s²)")

# Check if rod can fit in bowl
if L >= 2*R:
    st.error(f"❌ Rod too long! Maximum length for bowl radius {R} is {2*R:.2f}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("📊 Display Options")
show_derivation = st.sidebar.checkbox("Show Full Derivation", False)
show_animation = st.sidebar.checkbox("Animate Angle Sweep", True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📐 System Visualization")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw bowl (hemisphere)
    theta_bowl = np.linspace(np.pi, 2*np.pi, 100)
    x_bowl = R * np.cos(theta_bowl)
    y_bowl = R * np.sin(theta_bowl)
    ax.plot(x_bowl, y_bowl, 'b-', linewidth=3, label='Bowl Surface')
    
    # Calculate equilibrium angle
    # Geometry: For a rod in a bowl, the center of mass height is:
    # h(θ) = R - √(R² - (L/2)²·cos²θ) - (L/2)·sinθ
    # Minimize h by setting dh/dθ = 0
    
    # Symbolic derivation
    θ = sp.Symbol('θ', real=True)
    L_sym, R_sym = sp.symbols('L R', positive=True)
    
    # Center of mass height above bowl bottom
    # The rod's midpoint is at distance d from bowl center, where:
    # d² + (L/2)² = R²  (Pythagorean theorem for the triangle)
    # d = √(R² - (L/2)²)
    # But this is for horizontal rod. For angle θ:
    # The vertical position of COM relative to bowl center:
    h_com = R - sp.sqrt(R_sym**2 - (L_sym/2)**2) * sp.cos(θ) - (L_sym/2) * sp.sin(θ)
    
    # Minimize potential energy: dV/dθ = mg·dh/dθ = 0
    dh_dθ = sp.diff(h_com, θ)
    eq_solution = sp.solve(dh_dθ, θ)
    
    # Get the physical solution (between -π/2 and π/2)
    θ_eq = None
    for sol in eq_solution:
        try:
            val = float(sol.evalf(subs={L_sym: L, R_sym: R}))
            if -np.pi/2 < val < np.pi/2:
                θ_eq = val
                break
        except:
            continue
    
    if θ_eq is None:
        θ_eq = 0  # Default if no solution found
    
    # Calculate rod endpoints
    # Rod center is at angle θ_eq from horizontal
    d_from_center = np.sqrt(R**2 - (L/2)**2)  # Distance from bowl center to rod midpoint
    
    # Rod midpoint position
    x_mid = d_from_center * np.cos(θ_eq + np.pi/2)  # +π/2 because θ is from horizontal
    y_mid = -d_from_center * np.sin(θ_eq + np.pi/2)
    
    # Rod endpoints (perpendicular to radius at midpoint)
    # The rod is perpendicular to the line from bowl center to rod midpoint
    perp_angle = θ_eq + np.pi/2
    
    x1 = x_mid + (L/2) * np.cos(perp_angle)
    y1 = y_mid + (L/2) * np.sin(perp_angle)
    x2 = x_mid - (L/2) * np.cos(perp_angle)
    y2 = y_mid - (L/2) * np.sin(perp_angle)
    
    # Draw rod
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=4, label=f'Rod (θ = {θ_eq*180/np.pi:.1f}°)')
    
    # Draw center of mass
    ax.plot(x_mid, y_mid, 'go', markersize=12, label='Center of Mass', zorder=5)
    
    # Draw bowl center
    ax.plot(0, 0, 'k+', markersize=15, label='Bowl Center', zorder=5)
    
    # Draw reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Labels and styling
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    ax.set_title('Rod in Hemispherical Bowl - Equilibrium Position', 
                fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*0.2)
    
    # Add annotation for angle
    arc_radius = 0.3 * R
    arc_theta = np.linspace(0, θ_eq, 50)
    arc_x = arc_radius * np.cos(arc_theta)
    arc_y = arc_radius * np.sin(arc_theta)
    ax.plot(arc_x, arc_y, 'g--', linewidth=2)
    ax.text(arc_radius*0.7*np.cos(θ_eq/2), arc_radius*0.7*np.sin(θ_eq/2), 
            f'θ = {θ_eq*180/np.pi:.1f}°', fontsize=10, color='green')
    
    st.pyplot(fig)

with col2:
    st.subheader("📊 Equilibrium Results")
    
    # Calculate potential energy at equilibrium
    h_eq = R + y_mid  # Height above bowl bottom
    V_eq = m * g * h_eq
    
    st.metric("Equilibrium Angle (θ)", f"{θ_eq*180/np.pi:.2f}°")
    st.metric("COM Height above Bowl Bottom", f"{h_eq:.3f} m")
    st.metric("Potential Energy", f"{V_eq:.3f} J")
    
    st.markdown("---")
    
    # Stability check
    d2h_dθ2 = sp.diff(dh_dθ, θ)
    stability = d2h_dθ2.evalf(subs={θ: θ_eq, L_sym: L, R_sym: R})
    
    if stability > 0:
        st.success("✓ Stable Equilibrium (minimum potential energy)")
    else:
        st.warning("⚠️ Unstable Equilibrium (maximum potential energy)")
    
    st.markdown("---")
    st.info("""
    **Key Observations:**
    - Longer rods tend to lie flatter (smaller θ)
    - Larger bowls allow steeper angles
    - Mass and gravity don't affect equilibrium angle (only scale V)
    """)

# Detailed derivation section
if show_derivation:
    st.markdown("---")
    st.subheader("📚 Full Mathematical Derivation")
    
    st.markdown("""
    ### Step 1: Define the System
    
    **Generalized Coordinate**: θ = angle rod makes with horizontal
    
    **Geometry**:
    - Bowl radius: R
    - Rod length: L
    - Distance from bowl center to rod midpoint: d = √(R² - (L/2)²)
    """)
    
    st.latex(r"""
    \text{Rod midpoint position:}
    \begin{cases}
    x_{cm} = d \cdot \cos(\theta + \pi/2) \\
    y_{cm} = -d \cdot \sin(\theta + \pi/2)
    \end{cases}
    """)
    
    st.markdown("""
    ### Step 2: Potential Energy
    
    Height of center of mass above bowl bottom:
    """)
    
    st.latex(r"""
    h(\theta) = R + y_{cm} = R - d \cdot \sin(\theta + \pi/2)
    """)
    
    st.latex(r"""
    V(\theta) = m \cdot g \cdot h(\theta)
    """)
    
    st.markdown("""
    ### Step 3: Lagrangian (Statics)
    
    Since T = 0 (no motion):
    """)
    
    st.latex(r"""
    \mathcal{L} = T - V = -V
    """)
    
    st.markdown("""
    ### Step 4: Equilibrium Condition
    
    Minimize potential energy:
    """)
    
    st.latex(r"""
    \frac{\partial V}{\partial \theta} = 0 \quad \Rightarrow \quad \frac{\partial h}{\partial \theta} = 0
    """)
    
    st.latex(r"""
    \frac{dh}{d\theta} = -d \cdot \cos(\theta + \pi/2) = d \cdot \sin(\theta) = 0
    """)
    
    st.latex(r"""
    \Rightarrow \theta_{eq} = 0 \quad \text{(for symmetric bowl)}
    """)
    
    st.info("""
    **Note**: For a perfectly hemispherical bowl, the equilibrium is horizontal (θ = 0). 
    This makes physical sense - the center of mass is lowest when the rod is horizontal.
    """)

# Animation of angle sweep
if show_animation:
    st.markdown("---")
    st.subheader("🎬 Potential Energy vs Angle")
    
    # Calculate V(θ) for range of angles
    theta_range = np.linspace(-np.pi/2, np.pi/2, 200)
    V_values = []
    
    for θ_val in theta_range:
        d = np.sqrt(R**2 - (L/2)**2)
        y_com = -d * np.sin(θ_val + np.pi/2)
        h = R + y_com
        V = m * g * h
        V_values.append(V)
    
    V_values = np.array(V_values)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(theta_range * 180/np.pi, V_values, 'b-', linewidth=3, label='Potential Energy V(θ)')
    ax2.axvline(x=θ_eq * 180/np.pi, color='r', linestyle='--', linewidth=2, 
                label=f'Equilibrium: θ = {θ_eq*180/np.pi:.1f}°')
    ax2.axhline(y=V_eq, color='g', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax2.set_xlabel('Angle θ (degrees)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Potential Energy V (J)', fontsize=12, fontweight='bold')
    ax2.set_title('Potential Energy vs Rod Angle', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    
    st.pyplot(fig2)
    
    st.info("""
    **Interpretation**: The equilibrium angle corresponds to the **minimum** of the potential energy curve. 
    This is the essence of Lagrange's method for statics - find the configuration that minimizes V.
    """)

# Educational summary
st.markdown("---")
st.subheader("🎓 Learning Points")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    ### Why This Example?
    
    1. **Simple Geometry**: Easy to visualize and understand
    2. **Pure Statics**: T = 0, so L = -V (no kinetic energy complications)
    3. **Clear Physics**: Minimum potential energy = stable equilibrium
    4. **Single Coordinate**: Only one generalized coordinate (θ)
    
    ### Lagrange's Method Steps:
    
    1. Choose generalized coordinates (θ)
    2. Write kinetic energy T(θ, θ̇)
    3. Write potential energy V(θ)
    4. Form Lagrangian: L = T - V
    5. Apply Euler-Lagrange equation
    
    For statics: ∂V/∂θ = 0
    """)

with col_b:
    st.markdown("""
    ### Next Steps: Dynamics
    
    Once comfortable with statics, extend to dynamics:
    
    - Add time dependence: θ(t)
    - Include kinetic energy: T = ½Iθ̇²
    - Full Euler-Lagrange: d/dt(∂L/∂θ̇) = ∂L/∂θ
    - Solve differential equation for θ(t)
    - Analyze oscillations, stability, energy conservation
    
    ### Real-World Applications:
    
    - Mechanical engineering (linkages, suspensions)
    - Robotics (arm positioning, stability)
    - Structural engineering (arches, bridges)
    - Molecular physics (bond angles, conformations)
    """)

st.markdown("""
---
### 💡 Key Takeaway

**Lagrange's method transforms a complex force-balance problem into a simple energy minimization problem.** 

Instead of resolving forces and torques at contact points, we just:
1. Write the energy
2. Take derivatives
3. Solve algebraic equations

This is the power of analytical mechanics! 🚀
""")

