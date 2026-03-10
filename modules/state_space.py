# lib/state_space.py
import numpy as np
from scipy.linalg import eig

class MDOFSystem:
    """
    Models a Multi-Degree-of-Freedom spring-mass-damper system.
    Converts 2nd-order ODEs to 1st-order State-Space form.
    """
    
    def __init__(self, masses, stiffness, damping):
        """
        masses: array of masses [m1, m2, ...]
        stiffness: array of spring constants [k1, k2, ...] (including ground springs)
        damping: array of damping coefficients [c1, c2, ...]
        """
        self.n = len(masses)
        self.m = np.array(masses)
        self.k = np.array(stiffness)
        self.c = np.array(damping)
        
        # Build Mass, Damping, Stiffness Matrices
        self.M = np.diag(self.m)
        self.K = self._build_tridiagonal(self.k)
        self.C = self._build_tridiagonal(self.c)
        
        # Build State-Space Matrix A (2n x 2n)
        self.A = self._build_state_matrix()
        
        # Compute Eigenvalues
        self.eigenvalues, self.eigenvectors = eig(self.A)
    
    def _build_tridiagonal(self, values):
        """Builds tridiagonal matrix for chain system."""
        n = self.n
        matrix = np.zeros((n, n))
        for i in range(n):
            # Diagonal
            matrix[i, i] = values[i]
            if i < n - 1:
                matrix[i, i] += values[i + 1]
            # Off-diagonal
            if i > 0:
                matrix[i, i-1] = -values[i]
            if i < n - 1:
                matrix[i, i+1] = -values[i + 1]
        return matrix
    
    def _build_state_matrix(self):
        """
        Converts Mx'' + Cx' + Kx = 0 to state-space form:
        [x']   [  0       I  ] [x]
        [  ] = [            ] [ ]
        [v']   [-M⁻¹K  -M⁻¹C] [v]
        """
        n = self.n
        zero = np.zeros((n, n))
        eye = np.eye(n)
        Minv = np.linalg.inv(self.M)
        
        top = np.hstack([zero, eye])
        bottom = np.hstack([-Minv @ self.K, -Minv @ self.C])
        
        return np.vstack([top, bottom])
    
    def get_mode_shapes(self):
        """Extract mode shapes from eigenvectors (position components only)."""
        # Eigenvectors are 2n x 1, we want first n rows (positions)
        return self.eigenvectors[:self.n, :]
    
    def get_natural_frequencies(self):
        """Return natural frequencies in Hz from eigenvalues."""
        # Eigenvalues are complex: λ = σ + iω
        # ω_d = imag(λ), ω_n = |λ|
        omega_d = np.imag(self.eigenvalues)
        omega_n = np.abs(self.eigenvalues)
        return omega_n / (2 * np.pi)
    
    def get_damping_ratios(self):
        """Calculate damping ratio ζ from eigenvalues."""
        # ζ = -σ / |λ|
        sigma = np.real(self.eigenvalues)
        mag = np.abs(self.eigenvalues)
        return -sigma / mag