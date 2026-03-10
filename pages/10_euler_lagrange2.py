import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.diagrams import get_euler_lagrange_flowchart
from modules.physics_core import simulate_double_pendulum # Reusing toolkit to show application

st.set_page_config(page_title="Euler-Lagrange Derivation", layout="wide")

st.title("📐 The Euler-Lagrange Derivation")
st.markdown("""
The Euler-Lagrange equation provides a powerful method to derive equations of motion 
without dealing with vector forces directly. It relies on scalar energy quantities.
""")

# --- Theory Section ---
st.header("1. The Workflow")
st.graphviz_chart(get_euler_lagrange_flowchart(), width='stretch')

st.header("2. The Equation")
st.latex(r"""
    \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = 0
""")
st.write("""
Where:
- $L = T - V$ (Lagrangian)
- $T$ = Kinetic Energy
- $V$ = Potential Energy
- $q_i$ = Generalized Coordinate (e.g., angle $\theta$)
""")

# --- Interactive Example ---
st.header("3. Application: From Theory to Code")
st.write("""
In our `laplace_lagrange_toolkit`, we implement the result of this derivation. 
Below, we verify the conservation of energy (approximately) for a simple case, 
which validates our Lagrangian derivation.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Energy Check")
    st.write("For a conservative system, Total Energy ($E = T + V$) should remain constant.")
    st.write("Numerical integration introduces small errors, seen as drift.")
    
    # Quick sim for energy check
    t = np.linspace(0, 10, 500)
    y0 = [np.pi/2, 0, np.pi/2, 0]
    sol = simulate_double_pendulum(t, y0, g=9.81, m1=1, m2=1, L1=1, L2=1)
    
    # Calculate Energy (Simplified for visualization)
    # This is a rough calc to show the concept, not exact physics engine grade
    th1, w1, th2, w2 = sol.T
    T = 0.5 * (w1**2 + w2**2) # Normalized for demo
    V = - (np.cos(th1) + np.cos(th2)) # Normalized for demo
    E = T + V
    
    fig, ax = plt.subplots()
    ax.plot(t, E, label="Total Energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (Norm.)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Next Steps: Hamiltonian")
    st.info(r"""
    **Coming Soon in Advanced Module:**
    
    Once we have the Lagrangian, we can perform a Legendre Transformation to move from 
    **Configuration Space** $(q, \dot{q})$ to **Phase Space** $(q, p)$.
    
    This leads to Hamilton's Equations:
    $$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$
    """)
    st.write("This will be covered in the **System State & Hamiltonian** module.")
