# pages/7_🏗️_FEA_Connection.py
import streamlit as st
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="FEA Connection", layout="wide")
st.title("🏗️ From Spring-Mass to Finite Element Analysis")

st.markdown("""
The `lagrange_toolkit` uses the **same mathematical foundation** as professional FEA software.
The difference is scale and sparsity.
""")

# --- Comparison Tab ---
tab1, tab2, tab3 = st.tabs(["🔬 Your Approach", "🏭 Professional FEA", "🚀 Scaling Demo"])

with tab1:
    st.header("The Current Approach (Educational)")
    
    st.subheader("Strengths")
    st.markdown("""
    - ✅ **Transparent**: Every matrix entry is visible
    - ✅ **Educational**: Students see the physics directly
    - ✅ **Flexible**: Easy to modify for learning
    - ✅ **Sufficient**: Perfect for systems < 1,000 DOF
    """)
    
    st.subheader("Limitations")
    st.markdown("""
    - ❌ **Dense matrices**: O(N²) memory, O(N³) solve time
    - ❌ **Manual assembly**: Error-prone for complex topologies
    - ❌ **All eigenvalues**: Computes modes you don't need
    """)
    
    st.info("💡 **This is perfect for learning!** Real FEA adds complexity that would obscure the physics for students.")

with tab2:
    st.header("Professional FEA Workflow")
    
    st.graphviz_chart("""
    digraph FEA {
        rankdir=TB;
        node [shape=box, style="rounded,filled", fillcolor=white];
        
        CAD [label="1. CAD Geometry", fillcolor="#e3f2fd"];
        Mesh [label="2. Mesh Generation\n(10⁴-10⁷ elements)", fillcolor="#fff3e0"];
        Assemble [label="3. Sparse Matrix Assembly\n(O(N) memory)", fillcolor="#e8f5e9"];
        Solve [label="4. Iterative Sparse Solve\n(Krylov methods)", fillcolor="#fce4ec"];
        Post [label="5. Visualization\n(Stress, Mode Shapes)", fillcolor="#f3e5f5"];
        
        CAD -> Mesh -> Assemble -> Solve -> Post;
    }
    """, width='stretch')
    
    st.subheader("Key Differences")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Matrix Assembly**
        - Current code: Manual or NetworkX Laplacian
        - FEA: Element-by-element assembly with numerical integration
        """)
    with col2:
        st.markdown("""
        **Solvers**
        - This code: `np.linalg.eig` (dense, direct)
        - FEA: `eigsh`, ARPACK, SLEPc (sparse, iterative)
        """)
    
    st.subheader("Example: Beam Bending")
    st.markdown("""
    A continuous beam: $EI \\frac{d^4 w}{dx^4} = q(x)$
    
    Discretized into beam elements:
    - Each element has 2 nodes, 2 DOF per node (deflection + rotation)
    - Element stiffness matrix: 4×4
    - Global assembly: sparse, banded structure
    """)
    
    # Show a simple beam element matrix
    st.latex(r"""
        k^{(e)} = \frac{EI}{L^3} \begin{bmatrix}
            12 & 6L & -12 & 6L \\
            6L & 4L^2 & -6L & 2L^2 \\
            -12 & -6L & 12 & -6L \\
            6L & 2L^2 & -6L & 4L^2
        \end{bmatrix}
    """)
    
    st.caption("This 4×4 element matrix gets assembled into a huge sparse global matrix—exactly like the spring-mass K matrix, but for bending!")

with tab3:
    st.header("Scaling Demo: Dense vs Sparse")
    
    max_n = st.slider("Max System Size", 100, 5000, 2000)
    
    if st.button("🚀 Run Scaling Comparison"):
        import time
        
        sizes = np.linspace(100, max_n, 8, dtype=int)
        dense_times, sparse_times = [], []
        
        progress = st.progress(0)
        
        for i, n in enumerate(sizes):
            # Dense approach (your current method)
            start = time.time()
            K_dense = np.zeros((n, n))
            for j in range(n):
                K_dense[j, j] = 2
                if j > 0: K_dense[j, j-1] = -1
                if j < n-1: K_dense[j, j+1] = -1
            np.linalg.eigvalsh(K_dense)  # Find all eigenvalues
            dense_times.append(time.time() - start)
            
            # Sparse approach (FEA method)
            start = time.time()
            K_sparse = diags([2, -1, -1], [0, -1, 1], shape=(n, n), format='csr')
            eigsh(K_sparse, k=6, which='SM')  # Find only 6 eigenvalues
            sparse_times.append(time.time() - start)
            
            progress.progress((i + 1) / len(sizes))
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(sizes, dense_times, 'o-', label='Dense (Educatioma; Method)', linewidth=2)
        ax.loglog(sizes, sparse_times, 's--', label='Sparse (FEA Method)', linewidth=2)
        ax.loglog(sizes, 1e-7 * np.array(sizes)**3, 'r:', label='O(N³) Reference')
        ax.loglog(sizes, 1e-5 * np.array(sizes), 'g:', label='O(N) Reference')
        
        ax.set_xlabel('System Size (N DOF)')
        ax.set_ylabel('Computation Time (seconds)')
        ax.set_title('Dense vs Sparse Eigenvalue Solvers')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        st.pyplot(fig)
        
        st.success("""
        **Key Takeaway:** 
        - Dense: Works great for learning (< 1,000 DOF)
        - Sparse: Essential for real engineering (> 10,000 DOF)
        - Same physics, different numerical methods!
        """)

st.divider()

st.header("🎯 Where This Toolkit Fits")

st.markdown("""
| Application | Recommended Approach |
|------------|---------------------|
| **Learning Lagrangian/Hamiltonian** | ✅ The current dense approach |
| **Control system design (PID)** | ✅ Laplace + State-Space |
| **Modal analysis of small structures** | ✅ The eigenvalue solver |
| **Stress analysis of complex parts** | 🏭 Professional FEA (ANSYS, Abaqus) |
| **Large-scale dynamics (10⁴+ DOF)** | 🏭 Sparse solvers + parallel computing |
""")

st.info("""
**The Lagrange_Toolkit is the perfect bridge:** 
It teaches the *physics and mathematics* that underlie professional tools, 
without overwhelming students with numerical analysis complexity.

Once they understand *why* the matrices look the way they do, 
learning *how* to use ANSYS or COMSOL becomes much easier!
""")