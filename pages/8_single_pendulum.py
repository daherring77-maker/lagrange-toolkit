import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.integrate import solve_ivp
import time

# Page configuration
st.set_page_config(
    page_title="Lagrange: Physical Pendulum",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 Lagrangian Mechanics: Physical Pendulum")
st.markdown("""
### From Statics to Dynamics

Now we add **motion**! The rod can swing, so kinetic energy **T ≠ 0**. 

**System**: A uniform rod of length L and mass m, pivoted at one end, swinging under gravity.

**Key Transition**: 
- **Statics**: Minimize V (equilibrium)
- **Dynamics**: Solve Euler-Lagrange equations (motion over time)
""")

# Sidebar controls
st.sidebar.header("📐 System Parameters")

# Physical parameters
L = st.sidebar.slider("Rod Length (L)", 0.2, 3.0, 1.0, 0.1, 
                     help="Length of the pendulum rod (m)")
m = st.sidebar.slider("Rod Mass (m)", 0.1, 10.0, 2.0, 0.5,
                     help="Mass of the rod (kg)")
g = st.sidebar.slider("Gravity (g)", 1.0, 20.0, 9.81, 0.5,
                     help="Gravitational acceleration (m/s²)")

st.sidebar.markdown("---")
st.sidebar.header("🎬 Simulation Controls")

# Initial conditions
theta0_deg = st.sidebar.slider("Initial Angle (θ₀)", -170, 170, 30, 5,
                               help="Starting angle from vertical (degrees)")
theta0 = np.radians(theta0_deg)
omega0 = st.sidebar.slider("Initial Angular Velocity (ω₀)", -10.0, 10.0, 0.0, 0.5,
                           help="Starting angular velocity (rad/s)")

# Simulation parameters
t_max = st.sidebar.slider("Simulation Time (s)", 1, 20, 10, 1)
time_step = st.sidebar.slider("Time Step (ms)", 10, 200, 50, 10) / 1000.0

# Display options
st.sidebar.markdown("---")
st.sidebar.header("📊 Display Options")
show_derivation = st.sidebar.checkbox("Show Full Derivation", False)
show_comparison = st.sidebar.checkbox("Compare Linear vs Nonlinear", True)
show_phase_portrait = st.sidebar.checkbox("Show Phase Portrait", True)
show_energy_plot = st.sidebar.checkbox("Show Energy Plot", True)
show_trail = st.sidebar.checkbox("Show Trajectory Trail", True)

# Main content area
st.markdown("---")

# ============================================
# SECTION 1: System Visualization (Animation)
# ============================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎬 Pendulum Motion")
    
    # Create figure for animation
    fig_anim, (ax_rod, ax_phase) = plt.subplots(1, 2, figsize=(12, 6), 
                                                 gridspec_kw={'width_ratios': [1.5, 1]})
    
    # === Left: Pendulum animation ===
    ax_rod.set_xlim(-L*1.2, L*1.2)
    ax_rod.set_ylim(-L*1.3, L*0.3)
    ax_rod.set_aspect('equal')
    ax_rod.grid(True, alpha=0.3, linestyle='--')
    ax_rod.set_xlabel('x (m)', fontsize=10, fontweight='bold')
    ax_rod.set_ylabel('y (m)', fontsize=10, fontweight='bold')
    ax_rod.set_title('Physical Pendulum', fontsize=12, fontweight='bold')
    
    # Draw pivot point
    pivot = Circle((0, 0), 0.05*L, color='red', fill=True, zorder=10)
    ax_rod.add_patch(pivot)
    ax_rod.text(0.05, 0.05, 'Pivot', fontsize=9, color='red')
    
    # Draw rod (will be updated in animation)
    rod_line, = ax_rod.plot([], [], 'bo-', linewidth=4, markersize=10, label='Rod')
    
    # Draw center of mass marker
    com_marker, = ax_rod.plot([], [], 'go', markersize=12, label='Center of Mass', zorder=5)
    
    # Trajectory trail
    trail_length = 200
    trail_x, trail_y = [], []
    trail_line, = ax_rod.plot([], [], 'r-', alpha=0.3, linewidth=2, label='Trajectory')
    
    # Gravity arrow
    gravity_arrow = FancyArrowPatch((0.8*L, -0.5*L), (0.8*L, -1.0*L), 
                                    arrowstyle='->', color='purple', 
                                    linewidth=2, mutation_scale=20)
    ax_rod.add_patch(gravity_arrow)
    ax_rod.text(0.85*L, -0.75*L, 'g', fontsize=12, color='purple', fontweight='bold')
    
    ax_rod.legend(loc='upper right', fontsize=8)
    
    # === Right: Phase portrait ===
    if show_phase_portrait:
        ax_phase.set_xlim(-np.pi, np.pi)
        ax_phase.set_ylim(-10, 10)
        ax_phase.grid(True, alpha=0.3, linestyle='--')
        ax_phase.set_xlabel('θ (rad)', fontsize=10, fontweight='bold')
        ax_phase.set_ylabel('ω (rad/s)', fontsize=10, fontweight='bold')
        ax_phase.set_title('Phase Portrait (θ vs ω)', fontsize=12, fontweight='bold')
        
        # Grid lines at multiples of π
        for k in range(-3, 4):
            ax_phase.axvline(x=k*np.pi/2, color='gray', linestyle=':', alpha=0.3)
            if k != 0:
                ax_phase.text(k*np.pi/2, 9.5, f'{k}π/2', fontsize=8, ha='center')
        
        # Phase trajectory (will be updated)
        phase_line, = ax_phase.plot([], [], 'b-', linewidth=2, alpha=0.7)
        phase_point, = ax_phase.plot([], [], 'ro', markersize=8)
    
    plt.tight_layout()

with col2:
    st.subheader("📊 Equilibrium & System Info")
    
    # Moment of inertia for rod about end: I = (1/3)mL²
    I = (1/3) * m * L**2
    
    # Small-angle natural frequency: ω₀ = √(mgL/(2I))
    # For physical pendulum: ω₀ = √(mgd/I) where d = L/2
    omega_n = np.sqrt(m * g * (L/2) / I)
    period_small = 2 * np.pi / omega_n
    
    st.metric("Moment of Inertia (I)", f"{I:.4f} kg·m²")
    st.metric("Natural Frequency (ω₀)", f"{omega_n:.3f} rad/s")
    st.metric("Period (Small Angles)", f"{period_small:.3f} s")
    
    st.markdown("---")
    st.metric("Initial Angle", f"{theta0_deg:.1f}°")
    st.metric("Initial Angular Velocity", f"{omega0:.2f} rad/s")
    
    st.markdown("---")
    st.info("""
    **Physical Pendulum vs Simple Pendulum:**
    - Mass distributed along length
    - I = ⅓mL² (not mL²)
    - COM at L/2 (not at end)
    - Period depends on mass distribution
    """)

# ============================================
# SECTION 2: Derive Equations of Motion
# ============================================

st.markdown("---")

if show_derivation:
    st.subheader("📚 Full Mathematical Derivation")
    
    st.markdown("""
    ### Step 1: Define Generalized Coordinate
    
    **θ(t)** = angle from vertical (downward positive)
    
    ### Step 2: Kinetic Energy (T)
    
    For a rotating rigid body:
    """)
    
    st.latex(r"""
    T = \frac{1}{2} I \dot{\theta}^2
    """)
    
    st.markdown("""
    Moment of inertia of uniform rod about end:
    """)
    
    st.latex(r"""
    I = \frac{1}{3} m L^2
    """)
    
    st.latex(r"""
    \Rightarrow T = \frac{1}{2} \left( \frac{1}{3} m L^2 \right) \dot{\theta}^2 = \frac{1}{6} m L^2 \dot{\theta}^2
    """)
    
    st.markdown("""
    ### Step 3: Potential Energy (V)
    
    Height of center of mass above lowest point:
    """)
    
    st.latex(r"""
    h = \frac{L}{2} (1 - \cos\theta)
    """)
    
    st.latex(r"""
    V = m g h = m g \frac{L}{2} (1 - \cos\theta)
    """)
    
    st.markdown("""
    ### Step 4: Lagrangian
    
    """)
    
    st.latex(r"""
    \mathcal{L} = T - V = \frac{1}{6} m L^2 \dot{\theta}^2 - m g \frac{L}{2} (1 - \cos\theta)
    """)
    
    st.markdown("""
    ### Step 5: Euler-Lagrange Equation
    
    """)
    
    st.latex(r"""
    \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{\theta}} \right) = \frac{\partial \mathcal{L}}{\partial \theta}
    """)
    
    st.markdown("""
    Compute derivatives:
    """)
    
    st.latex(r"""
    \frac{\partial \mathcal{L}}{\partial \dot{\theta}} = \frac{1}{3} m L^2 \dot{\theta}
    \quad \Rightarrow \quad
    \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{\theta}} \right) = \frac{1}{3} m L^2 \ddot{\theta}
    """)
    
    st.latex(r"""
    \frac{\partial \mathcal{L}}{\partial \theta} = - m g \frac{L}{2} \sin\theta
    """)
    
    st.markdown("""
    Equate and simplify (divide by mL):
    """)
    
    st.latex(r"""
    \frac{1}{3} L \ddot{\theta} = - \frac{g}{2} \sin\theta
    """)
    
    st.latex(r"""
    \boxed{\ddot{\theta} + \frac{3g}{2L} \sin\theta = 0}
    """)
    
    st.markdown("""
    ### Step 6: Small-Angle Approximation
    
    For |θ| ≪ 1: sin θ ≈ θ
    """)
    
    st.latex(r"""
    \boxed{\ddot{\theta} + \frac{3g}{2L} \theta = 0}
    \quad \text{(Simple harmonic oscillator)}
    """)
    
    st.latex(r"""
    \text{Solution: } \theta(t) = \theta_0 \cos(\omega_0 t), \quad \omega_0 = \sqrt{\frac{3g}{2L}}
    """)
    
    st.info("""
    **Key Observations:**
    - Nonlinear equation: θ̈ + (3g/2L) sin θ = 0
    - Linear approximation valid for |θ| < ~15°
    - Natural frequency independent of mass!
    - Period increases with amplitude for large angles (nonlinear effect)
    """)

# ============================================
# SECTION 3: Solve Equations Numerically
# ============================================

st.markdown("---")
st.subheader("📈 Simulation Results")

# Define ODE system
def pendulum_ode_nonlinear(t, y):
    """Nonlinear pendulum: θ̈ + (3g/2L) sin θ = 0"""
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(3*g/(2*L)) * np.sin(theta)
    return [dtheta_dt, domega_dt]

def pendulum_ode_linear(t, y):
    """Linear pendulum: θ̈ + (3g/2L) θ = 0"""
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(3*g/(2*L)) * theta
    return [dtheta_dt, domega_dt]

# Time array
t_span = (0, t_max)
t_eval = np.linspace(0, t_max, int(t_max/time_step))

# Solve nonlinear system
sol_nonlinear = solve_ivp(pendulum_ode_nonlinear, t_span, [theta0, omega0], 
                          t_eval=t_eval, method='RK45')

# Solve linear system (if requested)
if show_comparison:
    sol_linear = solve_ivp(pendulum_ode_linear, t_span, [theta0, omega0], 
                           t_eval=t_eval, method='RK45')
    theta_lin = sol_linear.y[0]
    omega_lin = sol_linear.y[1]

# Extract solution
t = sol_nonlinear.t
theta_nl = sol_nonlinear.y[0]
omega_nl = sol_nonlinear.y[1]

# Calculate energies
I_val = (1/3) * m * L**2
T_energy = 0.5 * I_val * omega_nl**2
V_energy = m * g * (L/2) * (1 - np.cos(theta_nl))
E_total = T_energy + V_energy

# ============================================
# SECTION 4: Plot Results
# ============================================

# Angle vs Time plot
fig_angle, ax_angle = plt.subplots(figsize=(12, 5))

ax_angle.plot(t, np.degrees(theta_nl), 'b-', linewidth=2.5, 
              label='Nonlinear (sin θ)', alpha=0.9)
if show_comparison:
    ax_angle.plot(t, np.degrees(theta_lin), 'r--', linewidth=2, 
                  label='Linear (θ)', alpha=0.7)

ax_angle.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax_angle.set_ylabel('Angle θ (degrees)', fontsize=11, fontweight='bold')
ax_angle.set_title('Pendulum Angle vs Time', fontsize=13, fontweight='bold')
ax_angle.grid(True, alpha=0.3, linestyle='--')
ax_angle.legend(loc='best', fontsize=10)
ax_angle.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)

st.pyplot(fig_angle)

# Energy plot
if show_energy_plot:
    fig_energy, ax_energy = plt.subplots(figsize=(12, 5))
    
    ax_energy.plot(t, T_energy, 'g-', linewidth=2, label='Kinetic Energy (T)', alpha=0.8)
    ax_energy.plot(t, V_energy, 'r-', linewidth=2, label='Potential Energy (V)', alpha=0.8)
    ax_energy.plot(t, E_total, 'k-', linewidth=2.5, label='Total Energy (T+V)', alpha=0.9)
    
    ax_energy.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax_energy.set_ylabel('Energy (J)', fontsize=11, fontweight='bold')
    ax_energy.set_title('Energy Conservation', fontsize=13, fontweight='bold')
    ax_energy.grid(True, alpha=0.3, linestyle='--')
    ax_energy.legend(loc='best', fontsize=10)
    
    # Show energy conservation error
    energy_error = np.abs(E_total - E_total[0]) / E_total[0] * 100
    max_error = np.max(energy_error)
    
    st.pyplot(fig_energy)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.metric("Initial Total Energy", f"{E_total[0]:.4f} J")
    with col_e2:
        st.metric("Max Energy Error", f"{max_error:.4f}%")
    
    if max_error < 0.1:
        st.success("✓ Excellent energy conservation (numerical accuracy)")
    elif max_error < 1.0:
        st.warning("⚠️ Acceptable energy drift")
    else:
        st.error("❌ Significant energy drift - reduce time step")

# Phase portrait plot
if show_phase_portrait:
    fig_phase, ax_phase_static = plt.subplots(figsize=(8, 8))
    
    # Plot trajectory
    ax_phase_static.plot(theta_nl, omega_nl, 'b-', linewidth=2, alpha=0.8, 
                         label='Nonlinear Trajectory')
    if show_comparison:
        ax_phase_static.plot(theta_lin, omega_lin, 'r--', linewidth=2, alpha=0.7,
                             label='Linear Trajectory')
    
    # Mark starting point
    ax_phase_static.plot(theta_nl[0], omega_nl[0], 'go', markersize=10, 
                         label='Start', zorder=10)
    # Mark ending point
    ax_phase_static.plot(theta_nl[-1], omega_nl[-1], 'ro', markersize=10,
                         label='End', zorder=10)
    
    ax_phase_static.set_xlim(-np.pi, np.pi)
    ax_phase_static.set_ylim(-10, 10)
    ax_phase_static.grid(True, alpha=0.3, linestyle='--')
    ax_phase_static.set_xlabel('θ (rad)', fontsize=11, fontweight='bold')
    ax_phase_static.set_ylabel('ω (rad/s)', fontsize=11, fontweight='bold')
    ax_phase_static.set_title('Phase Portrait', fontsize=13, fontweight='bold')
    ax_phase_static.legend(loc='best', fontsize=10)
    
    # Grid lines at multiples of π
    for k in range(-3, 4):
        ax_phase_static.axvline(x=k*np.pi/2, color='gray', linestyle=':', alpha=0.3)
        if k != 0:
            ax_phase_static.text(k*np.pi/2, 9.5, f'{k}π/2', fontsize=9, ha='center')
    
    ax_phase_static.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_phase_static.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    st.pyplot(fig_phase)
    
    st.info("""
    **Phase Portrait Interpretation:**
    - **Elliptical orbits**: Periodic motion (libration)
    - **Separatrix**: Boundary between oscillation and rotation
    - **Center at (0,0)**: Stable equilibrium (hanging down)
    - **Saddle at (±π,0)**: Unstable equilibrium (inverted)
    - **Linear**: Perfect ellipses
    - **Nonlinear**: Distorted ellipses (flattened at top/bottom)
    """)

# ============================================
# SECTION 5: Real-time Animation
# ============================================

st.markdown("---")
st.subheader("🎬 Real-time Animation")

# Create animation function
def init_animation():
    rod_line.set_data([], [])
    com_marker.set_data([], [])
    trail_line.set_data([], [])
    if show_phase_portrait:
        phase_line.set_data([], [])
        phase_point.set_data([], [])
    return rod_line, com_marker, trail_line

def animate(frame):
    # Get current state
    theta_curr = theta_nl[frame]
    omega_curr = omega_nl[frame]
    
    # Calculate rod endpoints
    # Pivot at (0, 0), rod extends at angle θ from vertical
    # Vertical down is θ = 0, so position is at angle (θ - π/2) from x-axis
    angle_from_x = theta_curr - np.pi/2
    
    x_end = L * np.cos(angle_from_x)
    y_end = L * np.sin(angle_from_x)
    
    # Center of mass at L/2
    x_com = (L/2) * np.cos(angle_from_x)
    y_com = (L/2) * np.sin(angle_from_x)
    
    # Update rod
    rod_line.set_data([0, x_end], [0, y_end])
    
    # Update COM marker
    com_marker.set_data([x_com], [y_com])
    
    # Update trail
    if show_trail:
        trail_x.append(x_end)
        trail_y.append(y_end)
        if len(trail_x) > trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)
    
    # Update phase portrait
    if show_phase_portrait:
        phase_line.set_data(theta_nl[:frame+1], omega_nl[:frame+1])
        phase_point.set_data([theta_curr], [omega_curr])
    
    return rod_line, com_marker, trail_line

# Create animation
anim = FuncAnimation(fig_anim, animate, init_func=init_animation,
                     frames=len(t), interval=time_step*1000, blit=True)

# Display animation
anim_placeholder = st.empty()
anim_placeholder.pyplot(fig_anim)

# Add play controls
col_play1, col_play2, col_play3 = st.columns(3)

with col_play1:
    play_button = st.button("▶️ Play Animation")
with col_play2:
    pause_button = st.button("⏸️ Pause")
with col_play3:
    reset_button = st.button("⏹️ Reset")

if play_button:
    # Re-run animation
    trail_x.clear()
    trail_y.clear()
    for i in range(len(t)):
        animate(i)
        anim_placeholder.pyplot(fig_anim)
        time.sleep(time_step)

# ============================================
# SECTION 6: Educational Summary
# ============================================

st.markdown("---")
st.subheader("🎓 Learning Points")

col_learn1, col_learn2 = st.columns(2)

with col_learn1:
    st.markdown("""
    ### What We Learned
    
    1. **Kinetic Energy Matters**
       - T = ½Iθ̇² introduces inertia
       - Motion couples position and velocity
    
    2. **Euler-Lagrange Equation**
       - d/dt(∂L/∂θ̇) = ∂L/∂θ
       - Automatically handles constraints
    
    3. **Nonlinearity**
       - sin θ vs θ approximation
       - Period depends on amplitude
    
    4. **Energy Conservation**
       - T + V = constant (check numerical accuracy!)
       - Powerful verification tool
    
    5. **Phase Space**
       - Complete state description (θ, ω)
       - Reveals system topology
    """)

with col_learn2:
    st.markdown("""
    ### Next Steps: Double Pendulum
    
    Why this gets exciting:
    
    - **Two coupled nonlinear ODEs**
    - **Chaotic behavior** (sensitive to initial conditions)
    - **Same Lagrangian method** scales beautifully
    - **No additional complexity** in derivation process
    
    The pattern is now clear:
    1. Choose coordinates
    2. Write T and V
    3. Form L = T - V  
    4. Apply Euler-Lagrange
    5. Solve ODEs
    
    This works for **any** mechanical system!
    """)

st.markdown("""
---
### 💡 Key Takeaway

**Lagrange's method transforms dynamics into a systematic recipe:**

Instead of:
- Drawing free-body diagrams
- Resolving forces and torques  
- Applying Newton's laws at each point

We just:
1. Write energy expressions
2. Take derivatives
3. Solve resulting ODEs

**The complexity is in the physics (energy), not the mechanics (forces).** This is why Lagrangian mechanics powers everything from robotics to quantum field theory. 🚀

---
### 📊 Parameter Sensitivity

Try these experiments:
- **Large initial angle** (> 60°): See nonlinear effects
- **Zero initial velocity**: Pure oscillation
- **High initial velocity**: Rotation vs oscillation
- **Change L**: Period scales as √L
- **Change m**: No effect on motion (mass cancels!)
- **Change g**: Simulate Moon (g=1.6) vs Jupiter (g=24.8)

Each slider reveals a physical insight!
""")

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; font-size: 0.9em;'>
Built with ❤️ using Streamlit, SymPy, SciPy, and Matplotlib<br>
Part of the Lagrangian Mechanics Educational Toolkit
</div>
""", unsafe_allow_html=True)