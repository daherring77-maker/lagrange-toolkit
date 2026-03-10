# newton_page.py (revised)
import streamlit as st
import plotly.graph_objects as go
import numpy as np

#def newton_page():
st.title("🌌 Newton's Laws — And Their Limits")

# Dark-mode-safe LaTeX
st.markdown("""
<style>
.katex { font-size: 1.1em; }
.katex-display > .katex { 
    background-color: transparent !important; 
    color: #e0e0e0 !important; 
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
> *"Every action has an equal and opposite reaction."*  
> Simple. Powerful. **Brilliantly engineering-friendly.**  
> But as systems grow, Newton’s vector-based approach reveals a cost:  
> **Unknowns you don’t care about**, and **equations that explode in number.**
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚖️ The Core Idea", 
    "⛓️ Constraint Problem", 
    "📈 Scaling Pain",
    "🧠 Design Decisions"
])

with tab1:
    st.subheader("F = ma — In Its Full Glory")
    st.latex(r"\vec{F}_{\text{net}} = m \vec{a} = m \frac{d^2\vec{r}}{dt^2}")
    st.write(r"""
    - **Forces are vectors**: direction matters → 3 equations per particle (x, y, z).  
    - **Superposition**: total force = sum of all $\vec{F}_i$ (gravity, spring, thrust, friction…).  
    - **Intuitive for engineers**: thrusters, cables, hinges — all map cleanly to $\vec{F}$.
    """)

    # Interactive: Single particle under gravity + thrust
    st.markdown("#### 🛰️ Try It: A Particle in 2D")
    col_a, col_b = st.columns(2)
    with col_a:
        thrust_mag = st.slider("Thrust magnitude", 0.0, 20.0, 5.0)
        thrust_angle = st.slider("Thrust angle (°)", -180, 180, 45)
    with col_b:
        gravity = st.slider("Gravity (m/s²)", 0.0, 20.0, 9.81)

    theta = np.radians(thrust_angle)
    thrust_vec = np.array([thrust_mag * np.cos(theta), thrust_mag * np.sin(theta)])
    gravity_vec = np.array([0, -gravity])
    net_force = thrust_vec + gravity_vec

    fig = go.Figure()
    origin = np.array([0.0, 0.0])
    fig.add_trace(go.Scatter(x=[origin[0], thrust_vec[0]], y=[origin[1], thrust_vec[1]],
                            mode='lines+markers', name='Thrust',
                            line=dict(color='cyan', width=4),
                            marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[origin[0], gravity_vec[0]], y=[origin[1], gravity_vec[1]],
                            mode='lines+markers', name='Gravity',
                            line=dict(color='orange', width=4),
                            marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[origin[0], net_force[0]], y=[origin[1], net_force[1]],
                            mode='lines+markers', name='Net Force',
                            line=dict(color='lime', width=6, dash='dot'),
                            marker=dict(size=10)))

    fig.update_layout(
        title="Force Vectors (2D)",
        xaxis=dict(range=[-12, 12], zeroline=True),
        yaxis=dict(range=[-12, 12], zeroline=True),
        width=500, height=500,
        plot_bgcolor='rgba(30,30,30,1)',
        paper_bgcolor='rgba(20,20,20,1)',
        font_color='white'
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("⛓️ The Constraint Conundrum")
    st.write("""
    Imagine a **double pendulum**:  
    - Two masses, connected by rigid rods.  
    - Each mass feels **gravity** and **tension** from the rod.  

    In Newtonian form:
    - You must solve for **tension forces** (internal, unknown, not of interest!).  
    - 2 particles × 2D = 4 unknown accelerations  
    - Plus 2 tension magnitudes → **6 unknowns**, only 4 equations → underdetermined!  
    - Need **constraint equations** (e.g., rod length constant) → messy geometry.
    """)

    st.latex(r"\|\vec{r}_1\| = L_1,\quad \|\vec{r}_2 - \vec{r}_1\| = L_2")
    st.caption("Geometric constraints — simple to state, hard to embed in F=ma.")

    # Sketch: double pendulum with tension vectors
    st.markdown("##### 📐 Visual: Forces You *Must* Track")
    fig2 = go.Figure()
    L1, L2 = 1.0, 0.8
    theta1, theta2 = np.radians(45), np.radians(-30)
    x1, y1 = L1*np.sin(theta1), -L1*np.cos(theta1)
    x2, y2 = x1 + L2*np.sin(theta2), y1 - L2*np.cos(theta2)

    fig2.add_trace(go.Scatter(x=[0, x1, x2], y=[0, y1, y2], mode='markers+lines',
                                marker=dict(size=10, color=['white', 'red', 'blue']),
                                line=dict(color='gray', width=2),
                                name='Pendulum'))

    # Tension T1
    fig2.add_trace(go.Scatter(
        x=[0, -0.3*np.sin(theta1)], y=[0, 0.3*np.cos(theta1)],
        mode='lines', line=dict(color='yellow', width=3, dash='dash'),
        showlegend=False))
    fig2.add_annotation(x=-0.2*np.sin(theta1), y=0.2*np.cos(theta1), text="T₁", 
                        font=dict(color='yellow'), showarrow=False)

    # Tension T2
    fig2.add_trace(go.Scatter(
        x=[x1, x1 - 0.25*np.sin(theta2)], y=[y1, y1 + 0.25*np.cos(theta2)],
        mode='lines', line=dict(color='yellow', width=3, dash='dash'),
        showlegend=False))
    fig2.add_annotation(x=x1 - 0.15*np.sin(theta2), y=y1 + 0.15*np.cos(theta2), 
                        text="T₂", font=dict(color='yellow'), showarrow=False)

    fig2.update_layout(
        title="Double Pendulum: Tension Forces (Newtonian View)",
        xaxis=dict(range=[-1.5, 1.5], zeroline=False, showgrid=False),
        yaxis=dict(range=[-2.0, 0.5], zeroline=False, showgrid=False, scaleanchor="x", scaleratio=1),
        width=500, height=500,
        plot_bgcolor='rgba(30,30,30,1)',
        paper_bgcolor='rgba(20,20,20,1)',
        font_color='white'
    )
    st.plotly_chart(fig2, width='stretch')

    st.info("""
    🔑 **Key Insight**:  
    *T₁ and T₂ are not inputs — they’re reactions to motion.*  
    You’re solving for them *just to eliminate them later*.  
    **Lagrange avoids this entirely** — by never introducing them.
    """)

with tab3:
    st.subheader("📈 Scaling: From 1 to N Bodies")
    st.write("""
    | Particles | Newton (vector eqs) | Unknown Forces |
    |-----------|----------------------|----------------|
    | 1         | 3                    | 0 (if no constraints) |
    | 2 (free)  | 6                    | 0 |
    | 2 (linked)| 6 + 2 tensions       | 2 → need 2 constraints |
    | N (chain) | 3N                   | ~N constraint forces |
    """)
    
    st.markdown("""
    Now add:
    - Rotating reference frames → fictitious forces ($-2m\\vec{\\omega}\\times\\vec{v}$)
    - Non-Cartesian coordinates → basis vectors change → extra terms
    - Time-dependent constraints → Lagrange multipliers pile up

    > 🧠 **Engineer’s Summary**:  
    > *Newton gives you full control — at the cost of bookkeeping.*  
    > *Like writing COBOL with explicit memory addresses: powerful, but exhausting at scale.*
    """)

# ================================
# NEW: Tab 4 — Design Decisions
# ================================
with tab4:
    st.subheader("🧠 Why Lagrange? A Systems Perspective")
    st.write("""
    The move from Newton to Lagrange wasn’t just mathematical elegance —  
    it was a **design refactor** to improve maintainability, scalability, and clarity.

    Below are two key decisions that shaped analytical mechanics —  
    framed as if we were choosing an architecture for a new system.
    """)

    # Decision 1: Generalized Coordinates
    with st.expander("🔍 Design Decision: Why *qᵢ*, Not *x, y, z*?"):
        st.markdown(r"""
        **Problem**:  
        In Newtonian mechanics, changing coordinates (e.g., from Cartesian to polar) *rewrites all equations*. Forces, accelerations, constraints — everything is tangled with the representation.

        **Insight**:  
        Lagrange asked: *What if we treat the configuration space itself as the domain?*  
        → Let the *geometry of the system* define the variables — not the ambient space.

        **Trade-off**:  
        ✅ Gain: Equations become *coordinate-agnostic*. Add a pendulum? Just add $\theta$.  
        ❌ Cost: You lose direct force intuition — but gain *composability* (e.g., chain 10 pendulums with 10 $\theta_i$).

        > 💡 **Architect’s Note**:  
        > This is like moving from *procedural code* (hardwired x/y logic) to *object-oriented design* (a `Joint` class with `angle` property). Same system — better abstraction.
        """)

    # Decision 2: Ignoring Constraint Forces
    with st.expander("🔍 Design Decision: Why Never Compute Tension?"):
        st.markdown(r"""
        **Problem**:  
        In a double pendulum, 60% of the Newtonian effort goes into solving for tensions $ T_1, T_2 $ — forces the engineer *doesn’t control and doesn’t care about*.

        **Insight**:  
        Lagrange said: *If the constraint is holonomic (i.e., expressible as $f(q) = 0$), embed it in the coordinates — don’t fight it.*  
        → Use $\theta_1, \theta_2$; the rod lengths are *built in*.

        **Trade-off**:  
        ✅ Gain: Fewer equations, no spurious unknowns, automatic satisfaction of constraints.  
        ❌ Cost: Non-holonomic constraints (e.g., rolling without slipping) need extra machinery (Lagrange multipliers).

        > 💡 **Architect’s Note**:  
        > This is *encapsulation*: hide internal state (tension), expose interface (angles). Clients (controllers) only see what they need.
        """)

    st.divider()
    st.subheader("⏭️ What’s Next?")
    st.write("""
    We’ll see how the **Principle of Least Action** lets us:
    - Work with *scalars* (energy, not force),  
    - Choose *any coordinates* (angles, distances, curvatures),  
    - Let *constraints disappear* by design,  
    - And recover conservation laws from *symmetry* — no vectors needed.

    But first — a moment of respect:  
    > **Newton got us to the Moon.**  
    > Lagrange helps us *land softly*.
    """)

    if st.button("➡️ Proceed to Least Action", type="primary"):
        st.switch_page("pages/2_least_action.py")
        #st.session_state.current_page = "🎯 Least Action Intuition"
        #st.rerun()

# End of file