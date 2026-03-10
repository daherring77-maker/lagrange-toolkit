# lib/lagrange_points.py
import numpy as np
from scipy.optimize import minimize, root

class LagrangePointSolver:
    """
    Solves for Lagrange Points in the Circular Restricted Three-Body Problem (CR3BP).
    Uses scipy optimization to find stationary points of the effective potential.
    """
    
    def __init__(self, m1=1.0, m2=0.1, omega=1.0):
        """
        m1: Mass of primary body (e.g., Sun)
        m2: Mass of secondary body (e.g., Earth)
        omega: Angular velocity of rotating frame
        """
        self.m1 = m1
        self.m2 = m2
        self.omega = omega
        self.mu = m2 / (m1 + m2)  # Mass ratio
        self.total_mass = m1 + m2
        
        # Place masses on x-axis for simplicity
        self.r1 = -self.mu * 1.0  # Position of m1
        self.r2 = (1 - self.mu) * 1.0  # Position of m2
        
    def effective_potential(self, pos):
        """
        Calculates the effective potential (pseudo-potential) in the rotating frame.
        Φ_eff = -GM1/r1 - GM2/r2 - 0.5 * ω² * r²
        """
        x, y = pos
        r1_dist = np.sqrt((x - self.r1)**2 + y**2)
        r2_dist = np.sqrt((x - self.r2)**2 + y**2)
        r_sq = x**2 + y**2
        
        # Avoid division by zero
        eps = 1e-10
        r1_dist = max(r1_dist, eps)
        r2_dist = max(r2_dist, eps)
        
        potential = -(self.m1 / r1_dist) - (self.m2 / r2_dist) - 0.5 * self.omega**2 * r_sq
        return potential
    
    def gradient(self, pos):
        """Gradient of the effective potential (forces)."""
        x, y = pos
        r1_dist = np.sqrt((x - self.r1)**2 + y**2)
        r2_dist = np.sqrt((x - self.r2)**2 + y**2)
        
        eps = 1e-10
        r1_dist = max(r1_dist, eps)
        r2_dist = max(r2_dist, eps)
        
        dx = (self.m1 * (x - self.r1) / r1_dist**3) + \
             (self.m2 * (x - self.r2) / r2_dist**3) - \
             self.omega**2 * x
             
        dy = (self.m1 * y / r1_dist**3) + \
             (self.m2 * y / r2_dist**3) - \
             self.omega**2 * y
             
        return np.array([dx, dy])
    
    def find_lagrange_points(self):
        """
        Finds all 5 Lagrange points using scipy.optimize with multiple initial guesses.
        Returns dict with point names, positions, and stability info.
        """
        points = {}
        
        # Initial guesses for the 5 points
        guesses = {
            'L1': [0.5, 0.0],    # Between masses
            'L2': [1.2, 0.0],    # Beyond smaller mass
            'L3': [-1.2, 0.0],   # Beyond larger mass
            'L4': [0.5, 0.8],    # Triangular point
            'L5': [0.5, -0.8]    # Triangular point
        }
        
        for name, guess in guesses.items():
            result = minimize(
                lambda x: np.sum(self.gradient(x)**2),  # Minimize gradient magnitude
                guess,
                method='BFGS',
                options={'gtol': 1e-10}
            )
            points[name] = {
                'position': result.x,
                'potential': self.effective_potential(result.x),
                'converged': result.success
            }
        
        # Analyze stability
        points = self._analyze_stability(points)
        return points
    
    def _analyze_stability(self, points):
        """
        Linear stability analysis using Hessian of effective potential.
        Positive definite Hessian = stable (L4, L5)
        Indefinite Hessian = unstable (L1, L2, L3)
        """
        h = 1e-5
        for name, data in points.items():
            x, y = data['position']
            
            # Numerical Hessian
            f0 = self.effective_potential([x, y])
            fxx = (self.effective_potential([x+h, y]) - 2*f0 + self.effective_potential([x-h, y])) / h**2
            fyy = (self.effective_potential([x, y+h]) - 2*f0 + self.effective_potential([x, y-h])) / h**2
            fxy = (self.effective_potential([x+h, y+h]) - self.effective_potential([x+h, y-h]) - 
                   self.effective_potential([x-h, y+h]) + self.effective_potential([x-h, y-h])) / (4*h**2)
            
            hessian = np.array([[fxx, fxy], [fxy, fyy]])
            eigenvalues = np.linalg.eigvals(hessian)
            
            # Stability criterion for rotating frame
            is_stable = np.all(eigenvalues > 0)
            data['stable'] = is_stable
            data['eigenvalues'] = eigenvalues
            
        return points
    
    # Add these presets and improved guess logic to LagrangePointSolver class

    PRESETS = {
        "Demo (μ=0.1)": {"m1": 0.9, "m2": 0.1, "omega": 1.0, "mu": 0.1},
        "Earth-Sun (μ=3e-6)": {"m1": 1-3e-6, "m2": 3e-6, "omega": 1.0, "mu": 3e-6},
        "Earth-Moon (μ=0.012)": {"m1": 0.988, "m2": 0.012, "omega": 1.0, "mu": 0.012},
        "Binary Stars (μ=0.5)": {"m1": 0.5, "m2": 0.5, "omega": 1.0, "mu": 0.5},
}

def get_initial_guesses(self):
    """Generate smart initial guesses based on mass ratio."""
    mu = self.mu
    
    if mu < 0.01:  # Small mu: use Hill sphere approximations
        r_hill = (mu/3)**(1/3)
        return {
            'L1': [1 - mu - r_hill*0.95, 0.0],
            'L2': [1 - mu + r_hill*1.05, 0.0], 
            'L3': [-1.0 - (5/12)*mu, 0.0],
            'L4': [0.5 - mu, np.sqrt(3)/2],
            'L5': [0.5 - mu, -np.sqrt(3)/2],
        }
    elif mu < 0.3:  # Medium mu: interpolated guesses
        return {
            'L1': [0.7 - 0.3*mu, 0.0],
            'L2': [1.1 + 0.2*mu, 0.0],
            'L3': [-1.1 - 0.1*mu, 0.0], 
            'L4': [0.5 - mu, np.sqrt(3)/2 * (1 - 0.2*mu)],
            'L5': [0.5 - mu, -np.sqrt(3)/2 * (1 - 0.2*mu)],
        }
    else:  # Large mu (equal masses): symmetric guesses
        return {
            'L1': [0.0, 0.0],
            'L2': [1.5, 0.0],
            'L3': [-1.5, 0.0],
            'L4': [0.0, np.sqrt(3)/2],
            'L5': [0.0, -np.sqrt(3)/2],
        }

def find_lagrange_points(self, refine=True):
    """
    Finds all 5 Lagrange points with adaptive initial guesses.
    refine: If True, use root-finding for higher precision on collinear points.
    """
    points = {}
    guesses = self.get_initial_guesses()
    
    for name, guess in guesses.items():
        # First pass: minimize gradient magnitude
        result = minimize(
            lambda x: np.sum(self.gradient(x)**2),
            guess,
            method='BFGS',
            options={'gtol': 1e-12, 'maxiter': 1000}
        )
        
        # Optional refinement for collinear points using root-finding
        if refine and name in ['L1', 'L2', 'L3']:
            try:
                refined = root(
                    lambda x: self.gradient([x[0], 0.0]),  # Constrain to y=0
                    [result.x[0], 0.0],
                    method='hybr'
                )
                if refined.success:
                    result.x = refined.x
            except:
                pass  # Keep BFGS result if refinement fails
        
        points[name] = {
            'position': result.x,
            'potential': self.effective_potential(result.x),
            'converged': result.success,
            'initial_guess': guess
        }
    
    return self._analyze_stability(points)