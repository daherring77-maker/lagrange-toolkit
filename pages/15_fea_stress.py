# pages/8_🏗️_FEA_Stress_Analysis.py
import streamlit as st
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as tri

st.set_page_config(page_title="FEA Stress Analysis", layout="wide")
st.title("🏗️ FEA Stress Analysis: Cylinder Under Pressure")

st.markdown("""
**Goal:** Analyze stress in a thick-walled cylinder using Finite Element Analysis.
**Method:** 2D Axisymmetric formulation with linear triangular elements.
""")

# --- Parameters ---
st.sidebar.header("Cylinder Parameters")
inner_radius = st.sidebar.slider("Inner Radius (mm)", 10, 100, 50)
outer_radius = st.sidebar.slider("Outer Radius (mm)", 60, 200, 100)
length = st.sidebar.slider("Length (mm)", 50, 300, 100)
internal_pressure = st.sidebar.slider("Internal Pressure (MPa)", 1, 100, 10)

st.sidebar.header("Material Properties")
youngs_modulus = st.sidebar.number_input("Young's Modulus (GPa)", value=200.0)
poisson_ratio = st.sidebar.number_input("Poisson's Ratio", value=0.3)

st.sidebar.header("Mesh Settings")
n_radial = st.sidebar.slider("Radial Elements", 10, 100, 50)
n_axial = st.sidebar.slider("Axial Elements", 10, 100, 50)

# Calculate DOFs
n_nodes = (n_radial + 1) * (n_axial + 1)
n_dofs = n_nodes * 2  # 2 DOF per node (r, z)
st.sidebar.info(f"**Total DOFs:** {n_dofs:,}")

if n_dofs > 10000:
    st.sidebar.warning("⚠️ Large model - may take longer to solve")
else:
    st.sidebar.success("✅ Model size is manageable")

# --- Mesh Generation ---
st.header("1. Mesh Generation")

def generate_cylinder_mesh(r_inner, r_outer, length, n_rad, n_ax):
    """Generate structured mesh for cylinder cross-section."""
    nodes = []
    elements = []
    
    # Generate nodes
    for i in range(n_ax + 1):
        z = i * length / n_ax
        for j in range(n_rad + 1):
            r = r_inner + j * (r_outer - r_inner) / n_rad
            nodes.append([r, z])
    
    nodes = np.array(nodes)
    #nodes.shape == (N, 2)
    
    # Generate triangular elements (structured)
    for i in range(n_ax):
        for j in range(n_rad):
            # Node indices
            n1 = i * (n_rad + 1) + j
            n2 = n1 + 1
            n3 = (i + 1) * (n_rad + 1) + j
            n4 = n3 + 1
            
            # Two triangles per quad
            elements.append([n1, n2, n3])  # Lower triangle
            elements.append([n2, n4, n3])  # Upper triangle
    
    return nodes, np.array(elements)

nodes, elements = generate_cylinder_mesh(inner_radius, outer_radius, length, n_radial, n_axial)

st.write(f"Generated **{len(nodes)}** nodes and **{len(elements)}** elements")

# Visualize mesh
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(nodes[:, 0], nodes[:, 1], s=1, c='b', alpha=0.5)
#ax.triplot(nodes[:, 0], nodes[:, 1], elements, 'b-', linewidth=0.5, alpha=0.5)
ax.plot(nodes[:, 0], nodes[:, 1], 'r.', markersize=2)
ax.set_xlabel('Radius (mm)')
ax.set_ylabel('Axial Position (mm)')
ax.set_title('Cylinder Mesh (2D Axisymmetric Cross-Section)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- Element Stiffness Matrix ---
st.header("2. Element Formulation")

with st.expander("📐 View Element Stiffness Derivation"):
    st.latex(r"""
    k^{(e)} = \int_V B^T D B \, dV
    
    \text{Where:}
    - B = \text{Strain-displacement matrix}
    - D = \text{Constitutive matrix (plane stress)}
    """)
    
    st.markdown("""
    For axisymmetric problems, we integrate around the circumference:
    $$ k^{(e)} = 2\\pi \\int_A B^T D B \\, r \\, dA $$
    """)

def compute_element_stiffness_axisymmetric(coords, E, nu):
    """
    Compute stiffness matrix for linear triangular element (AXISYMMETRIC).
    
    Key differences from plane stress:
    1. 4 strain components (not 3): εr, εz, εθ (hoop), γrz
    2. D matrix is 4×4 (not 3×3)
    3. Integration includes 2πr for circumferential direction
    """
    r1, z1 = coords[0]
    r2, z2 = coords[1]
    r3, z3 = coords[2]
    
    # Element area
    A = 0.5 * ((r2*z3 - r3*z2) + (r3*z1 - r1*z3) + (r1*z2 - r2*z1))
    if A < 1e-10:
        raise ValueError(f"Degenerate element with area {A}")
    
    # Derivatives of shape functions (constant strain triangle)
    dN1_dr = (z2 - z3) / (2 * A)
    dN2_dr = (z3 - z1) / (2 * A)
    dN3_dr = (z1 - z2) / (2 * A)
    
    dN1_dz = (r3 - r2) / (2 * A)
    dN2_dz = (r1 - r3) / (2 * A)
    dN3_dz = (r2 - r1) / (2 * A)
    
    # Average radius for integration (and hoop strain)
    r_avg = (r1 + r2 + r3) / 3
    
    # Shape functions at centroid (for hoop strain εθ = u/r)
    N1 = N2 = N3 = 1/3
    
    # === B MATRIX (4×6) - AXISYMMETRIC ===
    # Strains: [εr, εz, εθ, γrz]
    # DOFs: [u1, v1, u2, v2, u3, v3]
    B = np.array([
        [dN1_dr, 0,        dN2_dr, 0,        dN3_dr, 0        ],  # εr = ∂u/∂r
        [0,      dN1_dz,   0,      dN2_dz,   0,      dN3_dz   ],  # εz = ∂v/∂z
        [N1/r_avg, 0,      N2/r_avg, 0,      N3/r_avg, 0      ],  # εθ = u/r (HOOP!)
        [dN1_dz, dN1_dr,  dN2_dz, dN2_dr,  dN3_dz, dN3_dr    ],  # γrz = ∂u/∂z + ∂v/∂r
    ])
    
    # === D MATRIX (4×4) - AXISYMMETRIC ===
    # For isotropic linear elastic material
    factor = E / ((1 + nu) * (1 - 2*nu))
    D = factor * np.array([
        [1-nu,   nu,    nu,    0     ],
        [nu,     1-nu,  nu,    0     ],
        [nu,     nu,    1-nu,  0     ],
        [0,      0,     0,     (1-2*nu)/2],
    ])
    
    # === ELEMENT STIFFNESS ===
    # k = ∫ B^T D B dV = B^T D B * (2π * r_avg * A)
    # The 2πr comes from integrating around the circumference
    volume = 2 * np.pi * r_avg * A
    k_elem = B.T @ D @ B * volume
    
    return k_elem

# --- Global Assembly ---
st.header("3. Global Matrix Assembly")

def assemble_global_stiffness(nodes, elements, E, nu):
    """Assemble global stiffness matrix (sparse)."""
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2
    
    # Use LIL format for efficient assembly
    K = lil_matrix((n_dofs, n_dofs))
    
    for elem in elements:
        # Get element coordinates
        elem_coords = nodes[elem]
        
        # Compute element stiffness
        k_elem = compute_element_stiffness_axisymmetric(elem_coords, E, nu)
        
        # Assemble into global matrix
        for i in range(3):
            for j in range(3):
                dof_i = [2*elem[i], 2*elem[i]+1]
                dof_j = [2*elem[j], 2*elem[j]+1]
                
                for ii in range(2):
                    for jj in range(2):
                        K[dof_i[ii], dof_j[jj]] += k_elem[2*i+ii, 2*j+jj]
    
    return csr_matrix(K)  # Convert to CSR for efficient solving

st.write("Assembling global stiffness matrix...")

with st.spinner("Computing element matrices and assembling..."):
    K_global = assemble_global_stiffness(nodes, elements, youngs_modulus * 1e9, poisson_ratio)

st.success(f"✅ Assembled {K_global.nnz:,} non-zero entries in {n_dofs}×{n_dofs} matrix")
st.write(f"**Sparsity:** {100*(1 - K_global.nnz/(n_dofs**n_dofs)):.4f}%")

# --- Boundary Conditions ---
st.header("4. Boundary Conditions")

def apply_boundary_conditions(K, nodes, elements, inner_radius, pressure):
    """
    Apply boundary conditions and pressure loads for axisymmetric cylinder.
    
    Parameters:
    - pressure: Internal pressure in Pa (positive = outward)
    - n_radial: Number of radial elements (for edge length calculation)
    """
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2
    F = np.zeros(n_dofs)
    
    # === 1. APPLY PRESSURE LOAD ON INNER SURFACE ===
    # Find all elements that have an edge on the inner radius
    inner_surface_edges = []
    
    for elem in elements:
        r1, z1 = nodes[elem[0]]
        r2, z2 = nodes[elem[1]]
        r3, z3 = nodes[elem[2]]
        
        # Check if any edge is on the inner surface
        tolerance = 0.01  # mm
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        
        for n1, n2 in edges:
            r_n1, z_n1 = nodes[n1]
            r_n2, z_n2 = nodes[n2]
            
            # Both nodes on inner radius = edge is on inner surface
            if abs(r_n1 - inner_radius) < tolerance and abs(r_n2 - inner_radius) < tolerance:
                # Avoid duplicate edges
                edge = tuple(sorted([n1, n2]))
                if edge not in inner_surface_edges:
                    inner_surface_edges.append(edge)
    
    # Apply pressure as nodal forces on inner surface edges
    for n1, n2 in inner_surface_edges:
        r1, z1 = nodes[n1]
        r2, z2 = nodes[n2]
        
        # Edge length
        L = np.sqrt((r2 - r1)**2 + (z2 - z1)**2)
        
        # Average radius for this edge (for 2πr integration)
        r_avg = (r1 + r2) / 2
        
        # Total force on this edge = pressure × area
        # Area = edge_length × circumference = L × 2πr_avg
        # Convert to mm for consistency: r_avg is in mm, L is in mm
        # Pressure is in Pa = N/m², so we need m² for area
        area_m2 = (L / 1000) * (2 * np.pi * r_avg / 1000)  # Convert mm to m
        total_force = pressure * area_m2  # Newtons
        
        # Distribute force equally to both nodes (linear elements)
        # Force acts in RADIAL direction (DOF 0, 2, 4, ... = even indices)
        force_per_node = total_force / 2
        
        # POSITIVE = outward (away from center)
        F[2 * n1] += force_per_node  # Radial DOF for node 1
        F[2 * n2] += force_per_node  # Radial DOF for node 2
    
    # === 2. APPLY BOUNDARY CONDITIONS ===
    # Fix bottom surface (z=0) in axial direction only
    # Allow radial expansion (Poisson effect)
    fixed_dofs = []
    for i, node in enumerate(nodes):
        r, z = node
        if abs(z) < 0.01:  # Bottom surface
            fixed_dofs.append(2 * i + 1)  # Fix z-DOF (odd indices)
    
    # Apply boundary conditions using penalty method
    K_modified = K.copy()
    penalty = 1e10
    for dof in fixed_dofs:
        K_modified[dof, dof] += penalty
    
    return K_modified, F

st.write("Applying boundary conditions:")
st.markdown("- Fixed bottom surface (axial constraint)")
st.markdown("- Internal pressure on inner radius")

K_bc, F_global = apply_boundary_conditions(K_global, nodes, elements, inner_radius, internal_pressure * 1e6)

# --- Solve ---
st.header("5. Solution")

with st.spinner("Solving system of equations..."):
    U = spsolve(K_bc, F_global)

st.success("✅ System solved!")

# Extract displacements
u_r = U[0::2]  # Radial displacements
u_z = U[1::2]  # Axial displacements
u_r = u_r * 1e6
u_z = u_z * 1e6
# --- Post-processing ---
st.header("6. Results Visualization")

# Deformed shape
deformed_r = nodes[:, 0] + u_r * 10  # Scale for visibility
deformed_z = nodes[:, 1] + u_z * 10

fig, ax = plt.subplots(figsize=(12, 6))

#ax.triplot(triangulation, 'gray', linewidth=0.5, alpha=0.3)
triang_original = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
triang_deformed = tri.Triangulation(deformed_r, deformed_z, elements)
# Plot mesh using triangulation object
ax.triplot(triang_original, color='gray', linewidth=0.5, alpha=0.3)
ax.triplot(triang_deformed, color='red', linewidth=0.5, alpha=0.7)
ax.plot(nodes[:, 0], nodes[:, 1], 'r.', markersize=2)
#ax.plot(nodes[:, 0], nodes[:, 1], 'b.', markersize=1, alpha=0.5)

ax.set_xlabel('Radius (mm)')
ax.set_ylabel('Axial Position (mm)')
ax.set_title('Cylinder Mesh (2D Axisymmetric Cross-Section)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

st.pyplot(fig)

#ax.plot(nodes[:, 0], nodes[:, 1], 'b.', markersize=1, alpha=0.5)
#ax.set_xlabel('Radius (mm)')
#ax.set_ylabel('Axial Position (mm)')
#ax.set_title('Deformed Shape (10x exaggeration)')
#ax.set_aspect('equal')
#ax.legend(['Original', 'Deformed'])
#ax.grid(True, alpha=0.3)
#st.pyplot(fig)

# Radial displacement contour
fig, ax = plt.subplots(figsize=(10, 6))
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
contour = ax.tricontourf(triang, u_r, levels=50, cmap='jet')
plt.colorbar(contour, ax=ax, label='Radial Displacement (mm)')
ax.set_xlabel('Radius (mm)')
ax.set_ylabel('Axial Position (mm)')
ax.set_title('Radial Displacement Contour')
ax.set_aspect('equal')
st.pyplot(fig)

# --- Analytical Comparison ---
# --- Analytical Comparison (Corrected) ---
st.header("7. Validation: Analytical Solution")

st.markdown("""
For a thick-walled cylinder under internal pressure (plane stress, axisymmetric):
""")

st.latex(r"""
u_r(r) = \frac{p_i r_i^2}{E(r_o^2 - r_i^2)} \left[ (1-\nu)r + (1+\nu)\frac{r_o^2}{r} \right]
""")

# Convert ALL inputs to SI units (meters, Pascals) for analytical formula
p_i_Pa = internal_pressure * 1e6  # MPa → Pa
r_i_m = inner_radius / 1000  # mm → m
r_o_m = outer_radius / 1000  # mm → m
E_Pa = youngs_modulus * 1e9 # GPa → Pa
nu = poisson_ratio

# Analytical solution in meters
r_values_m = np.linspace(r_i_m, r_o_m, 100)
u_analytical_m = (p_i_Pa * r_i_m**2 / (E_Pa * (r_o_m**2 - r_i_m**2))) * \
                 ((1-nu)*r_values_m + (1+nu)*r_o_m**2/r_values_m)

# Convert analytical result to mm for comparison
u_analytical_mm = u_analytical_m * 1000

# Extract FEA results at mid-height (same radial positions)
# Nodes are ordered: row by row (axial first, then radial)
mid_row = n_axial // 2  # Middle axial position
start_idx = mid_row * (n_radial + 1)  # First node in middle row
end_idx = start_idx + (n_radial + 1)  # Last node in middle row + 1

r_fem_mm = nodes[start_idx:end_idx, 0]  # Radial positions (already in mm)
u_fem_mm = u_r[start_idx:end_idx]  # Radial displacements (in mm)

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r_values_m * 1000, u_analytical_mm, 'r-', linewidth=2, label='Analytical')
ax.plot(r_fem_mm, u_fem_mm, 'bo', markersize=4, label='FEA', alpha=0.6)
ax.set_xlabel('Radius (mm)')
ax.set_ylabel('Radial Displacement (mm)')
ax.set_title('FEA vs Analytical Solution (Mid-Height)')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Error calculation (in mm)
u_analytical_interp = np.interp(r_fem_mm, r_values_m * 1000, u_analytical_mm)
error_mm = np.abs(u_fem_mm - u_analytical_interp)
max_error_mm = np.max(error_mm)
avg_error_mm = np.mean(error_mm)
relative_error_pct = np.mean(np.abs(error_mm / u_analytical_interp)) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Maximum Error", f"{max_error_mm:.6f} mm")
col2.metric("Average Error", f"{avg_error_mm:.6f} mm")
col3.metric("Relative Error", f"{relative_error_pct:.3f}%")

if relative_error_pct < 5:
    st.success(f"✅ Excellent agreement! FEA matches analytical within {relative_error_pct:.2f}%")
elif relative_error_pct < 15:
    st.warning(f"⚠️ Reasonable agreement. Consider refining mesh for better accuracy.")
else:
    st.error(f"❌ Large discrepancy. Check units, boundary conditions, or mesh quality.")

# --- Performance Stats ---
st.divider()
st.header("📊 Performance Statistics")

import sys
memory_mb = (K_global.data.nbytes + K_global.indices.nbytes + K_global.indptr.nbytes) / 1024 / 1024

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total DOFs", f"{n_dofs:,}")
col2.metric("Non-zero Entries", f"{K_global.nnz:,}")
col3.metric("Memory Usage", f"{memory_mb:.1f} MB")
col4.metric("Sparsity", f"{100*(1 - K_global.nnz/(n_dofs**2)):.2f}%")

st.info("""
**Key Takeaway:** 
Sparse matrices make large-scale FEA practical. 
A dense 10,000 DOF matrix would require ~800 MB, 
but sparse format uses only ~10-50 MB for typical FEA problems!
""")
# --- Debug Info ---
with st.expander("🔍 Debug: Check Units and Values"):
    st.write("**Input Values (as used in FEA):**")
    st.write(f"- Inner radius: {inner_radius} mm = {inner_radius/1000:.4f} m")
    st.write(f"- Outer radius: {outer_radius} mm = {outer_radius/1000:.4f} m")
    st.write(f"- Pressure: {internal_pressure} MPa = {internal_pressure*1e6:.0f} Pa")
    st.write(f"- Young's Modulus: {youngs_modulus} GPa = {youngs_modulus*1e9:.0f} Pa")
    
    st.write("**FEA Results (mid-height, inner surface):**")
    inner_node_idx = start_idx  # First node in middle row = inner radius
    st.write(f"- Radial position: {r_fem_mm[0]:.2f} mm")
    st.write(f"- FEA displacement: {u_fem_mm[0]:.6f} mm")
    
    st.write("**Analytical Prediction (same location):**")
    u_analytical_inner = (p_i_Pa * r_i_m**2 / (E_Pa * (r_o_m**2 - r_i_m**2))) * \
                         ((1-nu)*r_i_m + (1+nu)*r_o_m**2/r_i_m) * 1000  # Convert to mm
    st.write(f"- Analytical displacement: {u_analytical_inner:.6f} mm")
    
    st.write("**Ratio (FEA/Analytical):**")
    ratio = u_fem_mm[0] / u_analytical_inner if u_analytical_inner != 0 else float('inf')
    st.write(f"- {ratio:.4f} (should be ≈ 1.0)")
    
    if ratio > 1000 or ratio < 0.001:
        st.error("🚨 Units mismatch likely! Check E, pressure, and radii conversions.")