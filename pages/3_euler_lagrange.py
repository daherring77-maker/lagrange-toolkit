import streamlit as st
import sympy as sp
from sympy import Symbol, Function, Derivative, diff, simplify, Eq
import math

# Page Configuration
st.set_page_config(page_title="Lagrangian Mechanics Explorer", layout="centered")

# Custom CSS for better math rendering and layout
st.markdown("""
    <style>
    .math-equation {
        font-size: 1.2rem;
        font-family: 'Courier New', Courier, monospace;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .concept-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-left: 5px solid #1e88e5;
        margin: 10px 0;
    }
    .result-box {
        background-color: transparent !important;
        padding: 15px;
        border-left: 5px solid #00897b;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def render_math_latex(expr):
    return f"""
    <div class="math-equation">
        ${sp.latex(expr)}$
    </div>
    """

# --- INTRO SECTION ---
st.title("Deriving Euler-Lagrange: From $L = T - V$ to $F = ma$")
st.markdown("""
The goal of this tool is to demonstrate that the formula $L = T - V$ is not arbitrary. 
It is a specific definition that, when applied to the **Euler-Lagrange Equation**, 
generates Newton's Laws of Motion.
""")

# --- SECTION 1: THE ACTION PRINCIPLE ---
st.header("1. The Principle of Least Action")
st.markdown("""
The foundation of Lagrangian mechanics is the **Action** ($S$). 
The action is the integral of the Lagrangian over time.
""")

t = sp.Symbol('t')
st.latex("S = \\int_{t_1}^{t_2} L \\, dt")

st.info("""
> **The Principle:** A physical system will move along the path that makes the action $S$ stationary (usually a minimum).
> 
> To find the stationary path, we use the **Calculus of Variations**. This process yields the Euler-Lagrange Equation.
""")

# --- SECTION 2: DEFINING THE LAGRANGIAN ---
st.header("2. Why $L = T - V$?")
st.markdown("""
We define the Lagrangian $L$ as the **Kinetic Energy ($T$)** minus the **Potential Energy ($V$)**.
""")

# Define symbols for math
m = sp.Symbol('m', positive=True)
x = Function('x')(t)
t = sp.Symbol('t')

# Kinetic Energy: 1/2 m v^2
v = diff(x, t) # velocity is derivative of position
T = (1/2) * m * v**2

# Potential Energy: Generic V(x)
V = sp.Function('V')(x)

L = T - V
st.subheader("The Definition")
st.write("Kinetic Energy ($T$):")
st.latex("T = \\frac{1}{2} m \\dot{x}^2")
st.write("Potential Energy ($V$) - A function of position:")
st.latex("V = V(x)")
st.write("---")
st.subheader("The Lagrangian Definition")
st.write("We construct $L$ by subtracting potential from kinetic:")
st.latex("L(x, \\dot{x}, t) = T - V = \\frac{1}{2} m \\dot{x}^2 - V(x)")

st.info("""
Why subtract?        
If we added them ($T+V$), the resulting equation of motion would be $0 = -\\nabla V$, implying no motion. 
The subtraction creates the necessary "tug-of-war" where an increase in velocity increases $L$, but a steep potential gradient decreases $L$, leading to dynamic equilibrium (Motion).,
""")


# --- SECTION 3: DERIVING EULER-LAGRANGE ---
st.header("3. Deriving the Euler-Lagrange Equation")
st.markdown("To find the path $x(t)$ that minimizes $S$, we perturb the path by a small amount $\\eta(t)$ (eta).")

st.markdown("""
**Step 1: The Perturbed Path**
$$ x_{perturbed}(t) = x(t) + \\epsilon \\eta(t) $$
""")

st.markdown("""
**Step 2: The Derivative**
The velocity of the perturbed path is:
$$ \\dot{x}_{perturbed} = \\dot{x} + \\epsilon \\dot{\\eta}(t) $$
""")

st.markdown("""
**Step 3: Differentiating Action w.r.t. $\\epsilon$**
We require the derivative of Action $S$ with respect to the perturbation strength $\\epsilon$ to be zero at $\\epsilon = 0$:
""")

st.latex("\\frac{dS}{d\\epsilon} = \\int_{t_1}^{t_2} \\left( \\frac{\\partial L}{\\partial x} \\frac{dx}{d\\epsilon} + \\frac{\\partial L}{\\partial \\dot{x}} \\frac{d\\dot{x}}{d\\epsilon} \\right) dt = 0")

st.markdown("Substituting $\\frac{dx}{d\\epsilon} = \\eta$ and $\\frac{d\\dot{x}}{d\\epsilon} = \\dot{\\eta}$:")
st.latex("\\int_{t_1}^{t_2} \\left( \\frac{\\partial L}{\\partial x} \\eta + \\frac{\\partial L}{\\partial \\dot{x}} \\dot{\\eta} \\right) dt = 0")

st.markdown("""
**Step 4: Integration by Parts**
We integrate the second term by parts to remove $\\dot{\\eta}$:
$$ \\int u dv = uv - \\int v du $$
Let $u = \\frac{\\partial L}{\\partial \\dot{x}}, dv = \\dot{\\eta} dt$ 
$$ \\Rightarrow du = \\frac{d}{dt}\\left(\\frac{\\partial L}{\\partial \\dot{x}}\\right) dt, v = \\eta $$
""")

st.latex("\\int_{t_1}^{t_2} \\frac{\\partial L}{\\partial \\dot{x}} \\dot{\\eta} \\, dt = \\left[ \\frac{\\partial L}{\\partial \\dot{x}} \\eta \\right]_{t_1}^{t_2} - \\int_{t_1}^{t_2} \\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{x}} \\right) \\eta \\, dt")

st.markdown("""
**Step 5: The Boundary Condition**
We assume the perturbation $\\eta$ is zero at the start and end points ($\\eta(t_1) = \\eta(t_2) = 0$). 
Therefore, the boundary term $[ \\frac{\\partial L}{\\partial \\dot{x}} \\eta ]$ is **Zero**.
""")

st.markdown("""
**Step 6: The Final Equation**
Substituting back, we get:
$$ \\int_{t_1}^{t_2} \\left( \\frac{\\partial L}{\\partial x} - \\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{x}} \\right) \\right) \\eta \\, dt = 0 $$

Since $\\eta(t)$ is arbitrary (except at boundaries), the term inside the parentheses must be zero. 
This is the **Euler-Lagrange Equation**:
""")

st.markdown("---")
Euler_Lagrange_Eq = Eq( Derivative(L, x) - Derivative(Derivative(L, v), t), 0)
st.subheader("The Euler-Lagrange Equation")
st.latex("\\frac{\\partial L}{\\partial x} - \\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{x}} \\right) = 0")
st.markdown("---")

# --- SECTION 4: VERIFICATION (THE "DERIVATION" OF F=MA) ---
st.header("4. Verification: Getting $F = ma$ from $L = T - V$")

st.markdown("Now we plug our specific Lagrangian $L = \\frac{1}{2}m\\dot{x}^2 - V(x)$ into the Euler-Lagrange equation to see if it matches Newton's Law.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Part A: $\\partial L / \\partial x$")
    st.markdown("Differentiate $L$ with respect to position $x$. The kinetic energy term ($\\dot{x}^2$) disappears because it doesn't contain $x$.")
    # dL/dx
    dL_dx = diff(L, x)
    st.latex("\\frac{\\partial L}{\\partial x} = - \\frac{\\partial V}{\\partial x}")
    st.markdown(r"Recall that Force is defined as the negative gradient of potential energy: $F = -\frac{dV}{dx}$")
    st.latex("\\Rightarrow \\frac{\\partial L}{\\partial x} = F")

with col2:
    st.subheader("Part B: $d/dt (\\partial L / \\partial \\dot{x})$")
    st.markdown("Differentiate $L$ with respect to velocity $\\dot{x}$, then take the derivative with respect to time $t$.")    
    # dL/dv
    dL_dv = diff(L, v)
    st.latex("\\frac{\\partial L}{\\partial \\dot{x}} = m \\dot{x}")
    st.markdown("Now take the time derivative:")
    # d/dt (dL/dv)
    d_dt_dL_dv = diff(dL_dv, t)
    st.latex("\\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{x}} \\right) = m \\ddot{x}")
    st.markdown("Which is Mass times Acceleration ($ma$).")

st.subheader("Putting it all together")
st.markdown("Insert results from Part A and Part B into the Euler-Lagrange equation:")
st.latex("\\frac{\\partial L}{\\partial x} - \\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{x}} \\right) = 0")
st.latex("F - ma = 0")
st.markdown("---")
st.success("$$F = ma$$", icon="✅")
st.markdown(r"<div class='result-box'><h3>Conclusion</h3><p>We started with the <strong>Action Principle</strong>, defined <strong>$L = T - V$</strong>, derived the <strong>Euler-Lagrange Equation</strong> mathematically, and finally showed that this formalism reproduces <strong>Newton's Second Law</strong>. This confirms that <strong>$L = T - V</strong>$ is the correct formulation for classical mechanical systems.</p></div>", unsafe_allow_html=True)
