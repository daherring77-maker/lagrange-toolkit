# least_action_page.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad

#least_action_page():
st.title("🎯 The Principle of Least Action — Demystified")

st.markdown(r"""
> *“Nature is thrifty in all its actions.”* — Maupertuis (1744)  
>  
> But here’s the truth: **Nature isn’t minimizing — it’s *stationarizing*.**  
> And **Action** isn’t mystical — it’s just $\int (\text{Kinetic} - \text{Potential}) \, dt$.  
> Let’s make it concrete.
""")

# Dark-mode LaTeX fix
st.markdown("""
<style>
.katex { font-size: 1.1em; }
.katex-display > .katex { 
    background-color: transparent !important; 
    color: #e0e0e0 !important; 
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📛 The Name Problem", 
    "🧮 What *Is* Action?",
    "🔍 Interactive Path Explorer",
    "🧠 Design Decision"
])

with tab1:
    st.subheader("📛 ‘Least Action’ is a Misnomer")
    st.write(r"""
    - ✅ **True**: The *actual* path makes the **action stationary** — i.e., $delta S = 0$.  
    - ❌ **False**: It’s always the *minimum*.  
        → It can be a *maximum* or *saddle point* (e.g., light in elliptical mirrors).
    """)

    st.latex(r"\delta S = \delta \int_{t_1}^{t_2} L \, dt = 0 \quad \text{where } L = T - V")

    st.info("""
    > 💡 Think of it like a ball on a hill:  
    > - **Minimum**: valley (stable equilibrium)  
    > - **Maximum**: hilltop (unstable)  
    > - **Saddle**: mountain pass  
    > Nature chooses *any* point where the slope is zero — not just valleys.
    """)

    # Visual: S vs path parameter
    phi = np.linspace(-2, 2, 200)
    S_vals = phi**3 - 3*phi  # cubic: min, max, saddle
    opt_idx = np.argmin(np.abs(np.gradient(S_vals)))  # near stationary

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi, y=S_vals, mode='lines', name='Action S(φ)',
                            line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=[phi[opt_idx]], y=[S_vals[opt_idx]], 
                            mode='markers', name='Stationary point',
                            marker=dict(color='lime', size=12)))
    fig.update_layout(
        title="Action Can Be Min, Max, or Saddle",
        xaxis_title="Path parameter φ",
        yaxis_title="Action S",
        plot_bgcolor='rgba(30,30,30,1)',
        paper_bgcolor='rgba(20,20,20,1)',
        font_color='white'
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("🧮 What *Is* Action? Let’s Compute It.")
    st.write(r"""
    For a particle of mass $m$, moving in 1D under gravity:
    - Kinetic energy: $T = \frac{1}{2} m dot{x}^2$  
    - Potential energy: $V = m g x$  
    - Lagrangian: $L = T - V = \frac{1}{2} m dot{x}^2 - m g x$  
    - **Action**: $S = int_{t_0}^{t_f} L , dt$
    """)

    st.markdown("#### 📏 Try It: Compute $ S $ for a Simple Path")
    col1, col2 = st.columns(2)
    with col1:
        m = st.slider("Mass (kg)", 0.1, 5.0, 1.0)
        g = st.slider("Gravity (m/s²)", 0.0, 20.0, 9.81)
        t_f = st.slider("Time interval (s)", 0.5, 3.0, 1.0)
    with col2:
        # Assume path: x(t) = a*t + b*t² (quadratic)
        a = st.slider("Linear coeff (a)", -5.0, 5.0, 0.0)
        b = st.slider("Quadratic coeff (b)", -5.0, 5.0, -4.9)

    # Define path: x(t) = a*t + b*t^2
    def x(t): return a*t + b*t**2
    def dxdt(t): return a + 2*b*t
    def L(t): return 0.5*m*dxdt(t)**2 - m*g*x(t)

    # Compute action via numerical integration
    S_val, _ = quad(L, 0, t_f)

    st.latex(fr"S = \int_{{0}}^{{{t_f}}} \left[ \frac{{1}}{{2}}({m})({dxdt(0.5):.1f} + \dots)^2 - ({m})({g})({x(0.5):.1f} + \dots) \right] dt = {S_val:.2f} \ \text{{J·s}}")
    st.caption("Unit: Joule-seconds (same as Planck’s constant — not a coincidence!)")

    # Plot path and L(t)
    t_vals = np.linspace(0, t_f, 100)
    x_vals = x(t_vals)
    L_vals = [L(t) for t in t_vals]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_vals, y=x_vals, mode='lines', name='x(t)',
                                line=dict(color='cyan')))
    fig2.add_trace(go.Scatter(x=t_vals, y=L_vals, mode='lines', name='L(t)',
                                line=dict(color='orange', dash='dot')))
    fig2.update_layout(
        title=f"Path & Lagrangian (S = {S_val:.2f} J·s)",
        xaxis_title="Time (s)",
        yaxis_title="Position / L",
        plot_bgcolor='rgba(30,30,30,1)',
        paper_bgcolor='rgba(20,20,20,1)',
        font_color='white'
    )
    st.plotly_chart(fig2, width='stretch')

with tab3:
    st.subheader("🔍 Interactive: Vary the Path, Watch Action Change")
    st.write("""
    Imagine a ball thrown upward. The *true* path is a parabola (under gravity).  
    But what if it took a *different* path between the same start/end points?  
    Let’s compare **Action** for 3 paths:
    """)

    # Fixed endpoints: t=0→1s, x=0→0 (launched and caught at same height)
    t0, tf = 0.0, 1.0
    x0, xf = 0.0, 0.0
    m, g = 1.0, 9.81

    # Path 1: True path — parabola x = v0 t - 0.5 g t^2, with v0 = g*tf/2 = 4.905
    v0 = g * tf / 2
    def x_true(t): return v0*t - 0.5*g*t**2
    def L_true(t): 
        dx = v0 - g*t
        return 0.5*m*dx**2 - m*g*x_true(t)

    # Path 2: Straight line (unphysical)
    def x_line(t): return x0 + (xf - x0) * t / tf
    def L_line(t): 
        dx = (xf - x0) / tf
        return 0.5*m*dx**2 - m*g*x_line(t)

    # Path 3: "Lazy" path — slow start, fast end
    def x_lazy(t): return 4 * x0 * (1 - t/tf)**2 * (t/tf)  # cubic, zero at ends
    def L_lazy(t): 
        dx = 4 * x0 * ((1 - 2*t/tf)*(t/tf) + (1 - t/tf)**2) / tf
        return 0.5*m*dx**2 - m*g*x_lazy(t)

    # Compute actions
    S_true, _ = quad(L_true, t0, tf)
    S_line, _ = quad(L_line, t0, tf)
    S_lazy, _ = quad(L_lazy, t0, tf)

    st.markdown(f"""
    | Path | Description | Action $ S $ (J·s) |
    |------|-------------|---------------------|
    | 🟢 **True** | Parabolic (Newton’s solution) | `{S_true:.3f}` |
    | 🔴 Straight | Constant velocity | `{S_line:.3f}` |
    | 🔵 Lazy | Slow start, fast end | `{S_lazy:.3f}` |
    """)

    # Plot all paths
    t_plot = np.linspace(t0, tf, 100)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t_plot, y=[x_true(t) for t in t_plot],
                                mode='lines', name='True path (stationary S)',
                                line=dict(color='lime', width=3)))
    fig3.add_trace(go.Scatter(x=t_plot, y=[x_line(t) for t in t_plot],
                                mode='lines', name='Straight (higher S)',
                                line=dict(color='red', dash='dash')))
    fig3.add_trace(go.Scatter(x=t_plot, y=[x_lazy(t) for t in t_plot],
                                mode='lines', name='Lazy (higher S)',
                                line=dict(color='blue', dash='dot')))

    fig3.update_layout(
        title="Paths Between Same Endpoints — Only One Has Stationary Action",
        xaxis_title="Time (s)",
        yaxis_title="Height x(t)",
        plot_bgcolor='rgba(30,30,30,1)',
        paper_bgcolor='rgba(20,20,20,1)',
        font_color='white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig3, width='stretch')

    st.success(r"""
    ✅ The **true path** has *lower* action than these alternatives — but crucially,  
    if we perturb it *slightly*, $S$ doesn’t change (to first order). That’s $delta S = 0$.
    """)

with tab4:
    st.subheader("🧠 Design Decision: Why Optimize *Action*, Not Force?")
    with st.expander("🔍 Why Replace Newton with a Variational Principle?"):
        st.markdown("""
        **Problem**:  
        Newton’s laws are *local* (instantaneous forces) and *vectorial* — hard to generalize to fields, relativity, or quantum mechanics.

        **Insight**:  
        Maupertuis & Euler asked: *What if physics is fundamentally global?*  
        → Instead of “What force acts *now*?”, ask:  
        *“Which path, over time, makes this integral stationary?”*

        **Trade-off**:  
        ✅ Gain: Unified framework for particles, fields, light, QM (path integrals!).  
        ✅ Symmetries → conservation laws (Noether's theorem) emerge *automatically*.  
        ❌ Cost: Less intuitive for impulsive events (collisions) — though workarounds exist.

        > 💡 **Architect’s Note**:  
        > This is like replacing *procedural event handlers* (“onForce → update acceleration”)  
        > with a *declarative optimization layer* (“find path minimizing effort”).  
        > Same behavior — radically different design.
        """)

st.divider()
st.subheader("⏭️ Next: From Principle to Equations")
st.write(r"""
Now that we see *why* $delta S = 0 $, we’ll derive **Lagrange’s equations** —  
the practical engine that turns this principle into solvable ODEs.

Spoiler: It’s just calculus of variations + product rule.  
And once we have it — the double pendulum, robotic arms, and lunar landers await.

> 🌟 *Fun fact*: The same math describes light (Fermat), particles (Lagrange),  
> and even spacetime (Einstein’s Hilbert action). One principle — many worlds.
""")

if st.button("➡️ Derive Lagrange’s Equations", type="primary"):
    st.switch_page("pages/3_euler_lagrange.py")
    #st.session_state.current_page = "⚖️ Lagrange Derivation"
    #st.rerun()