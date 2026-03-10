# lib/physics_core.py
import numpy as np
from scipy.integrate import odeint

def double_pendulum_derivs(state, t, g, m1, m2, L1, L2):
    """
    Equations of motion for a double pendulum derived via Euler-Lagrange.
    state = [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = state

    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)

    # Denominators
    den1 = (m1 + m2) * L1 - m2 * L1 * c**2
    den2 = (L2 / L1) * den1

    # Accelerations
    alpha1 = (m2 * L1 * omega1**2 * s * c + 
              m2 * g * np.sin(theta2) * c + 
              m2 * L2 * omega2**2 * s - 
              (m1 + m2) * g * np.sin(theta1)) / den1
              
    alpha2 = (-(m2 * L2 * omega2**2 * s * c) + 
              (m1 + m2) * g * np.sin(theta1) * c - 
              (m1 + m2) * L1 * omega1**2 * s - 
              (m1 + m2) * g * np.sin(theta2)) / den2

    return [omega1, alpha1, omega2, alpha2]

def simulate_double_pendulum(t, y0, g=9.81, m1=1.0, m2=1.0, L1=1.0, L2=1.0):
    sol = odeint(double_pendulum_derivs, y0, t, args=(g, m1, m2, L1, L2))
    return sol