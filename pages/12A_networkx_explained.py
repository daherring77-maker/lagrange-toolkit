# pages/6_🔗_NetworkX_Explained.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="NetworkX Explained", layout="wide")
st.title("🔗 How NetworkX Connects to Differential Equations")

st.markdown("""
**NetworkX doesn't solve physics—it builds the system structure.**
The graph topology becomes your stiffness matrix, which goes into your ODE.
""")

# --- Example: 4-Mass System ---
st.header("Step-by-Step: From Graph to Equations")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. The Physical System")
    st.markdown("4 masses connected by springs:")
    
    # Draw physical system
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Wall
    ax.plot([0, 0], [-0.5, 0.5], 'k-', linewidth=3)
    ax.text(-0.3, 0, 'Wall', ha='right')
    
    # Masses
    positions = [1, 2, 3, 4]
    for i, pos in enumerate(positions):
        ax.add_patch(plt.Rectangle((pos-0.3, -0.3), 0.6, 0.6, 
                                    fill=True, color='lightblue', 
                                    edgecolor='black', linewidth=2))
        ax.text(pos, 0.5, f'm{i+1}', ha='center')
    
    # Springs
    spring_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
    for pos in spring_positions:
        ax.plot([pos-0.4, pos+0.4], [0, 0], 'k-', linewidth=1)
        ax.plot([pos-0.4, pos-0.2], [-0.1, 0.1], 'k-', linewidth=1)
        ax.plot([pos-0.2, pos], [0.1, -0.1], 'k-', linewidth=1)
        ax.plot([pos, pos+0.2], [-0.1, 0.1], 'k-', linewidth=1)
        ax.plot([pos+0.2, pos+0.4], [0.1, -0.1], 'k-', linewidth=1)
    
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.subheader("2. The Graph Representation")
    
    # Create equivalent graph
    G = nx.Graph()
    G.add_node(0, label='Wall')  # Ground
    for i in range(4):
        G.add_node(i+1, label=f'm{i+1}')
    
    # Edges = Springs
    G.add_edge(0, 1)  # Wall to m1
    for i in range(1, 4):
        G.add_edge(i, i+1)  # m1-m2, m2-m3, m3-m4
    
    # Draw graph
    fig, ax = plt.subplots(figsize=(6, 3))
    pos = {i: (i*0.8, 0) for i in range(5)}
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, ax=ax, edge_color='gray', width=2)
    st.pyplot(fig)

st.divider()

# --- Matrix Assembly ---
st.header("3. From Graph to Matrices")

st.markdown("""
NetworkX can extract the **adjacency** and **Laplacian** matrices from the graph.
These become your **Stiffness Matrix (K)** in the equation:
""")

st.latex(r"M\ddot{x} + Kx = 0")

# Show the actual matrices
n_masses = 4
G = nx.path_graph(n_masses + 1)  # Path graph = chain of masses

# Laplacian matrix (represents connectivity/stiffness)
L = nx.laplacian_matrix(G).toarray()

# Remove ground node (first row/col) to get mass system only
K = L[1:, 1:] + np.eye(n_masses) * 0.1  # Add small diagonal for stability

st.subheader("Stiffness Matrix K (from Graph Laplacian)")
st.write("Each row = force balance on one mass")
st.write("Diagonal = springs connected to this mass")
st.write("Off-diagonal = coupling between masses")

st.dataframe(np.round(K, 2), width='stretch')

st.info("""
**Key Insight:** 
- K[i,i] = number of springs connected to mass i
- K[i,j] = -1 if masses i and j are connected, 0 otherwise
- This is exactly what you'd write from F = ma for each mass!
""")

st.divider()

# --- Eigenvalues to Solution ---
st.header("4. From Matrices to Differential Equation Solution")

st.markdown("""
Once we have K and M, we solve the **eigenvalue problem**:
""")

st.latex(r"(K - \omega^2 M)\phi = 0")

st.markdown("""
The eigenvalues give us **natural frequencies**, and eigenvectors give us **mode shapes**.
The full time solution is:
""")

st.latex(r"x(t) = \sum_i c_i \phi_i \cos(\omega_i t + \psi_i)")

# Compute and show eigenvalues
M = np.eye(n_masses)
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)
frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

st.subheader("Natural Frequencies (from Eigenvalues)")
for i, freq in enumerate(sorted(frequencies)):
    st.write(f"**Mode {i+1}:** {freq:.3f} Hz")

# Plot mode shapes
st.subheader("Mode Shapes (Eigenvectors)")
fig, ax = plt.subplots(figsize=(8, 4))
for i in range(min(3, len(eigenvectors))):
    mode = np.real(eigenvectors[:, i])
    mode = mode / np.max(np.abs(mode))
    ax.plot(range(1, n_masses+1), mode, 'o-', linewidth=2, label=f'Mode {i+1}')

ax.set_xlabel('Mass Number')
ax.set_ylabel('Relative Displacement')
ax.set_title('First 3 Mode Shapes')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xticks(range(1, n_masses+1))
st.pyplot(fig)

st.divider()

# --- The Complete Pipeline ---
st.header("5. The Complete Pipeline")

st.graphviz_chart("""
digraph Pipeline {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica"];
    edge [fontname="Helvetica", fontsize=10];
    
    Graph [label="1. NetworkX Graph\n(Nodes=Masses, Edges=Springs)", fillcolor="#e3f2fd"];
    Laplacian [label="2. Laplacian Matrix\n(Graph Connectivity)", fillcolor="#fff3e0"];
    Stiffness [label="3. Stiffness Matrix K\n(Physics Assembly)", fillcolor="#e8f5e9"];
    Eigen [label="4. Eigenvalue Problem\n(K - ω²M)φ = 0", fillcolor="#fce4ec"];
    Solution [label="5. Time Solution\nx(t) = Σ φᵢ cos(ωᵢt)", fillcolor="#f3e5f5"];
    
    Graph -> Laplacian -> Stiffness -> Eigen -> Solution;
}
""", width='stretch')

st.divider()

# --- Why This Matters ---
st.header("6. Why Use NetworkX Instead of Manual Matrices?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Advantages")
    st.markdown("""
    - **Automatic assembly**: No manual matrix entry errors
    - **Complex topologies**: Easy to model trusses, frames, networks
    - **Graph algorithms**: Find critical paths, connected components
    - **Visualization**: Built-in graph drawing
    - **Scalability**: Same code works for 10 or 10,000 masses
    """)

with col2:
    st.subheader("⚠️ Trade-offs")
    st.markdown("""
    - **Abstraction layer**: Harder to see the physics directly
    - **Overhead**: Slightly slower for simple systems
    - **Learning curve**: Need to understand graph theory basics
    - **Debugging**: Matrix errors harder to trace to graph
    """)

st.info("""
**Best Practice:** 
- Use **manual matrices** for learning and simple systems (2-5 DOF)
- Use **NetworkX** for complex structures (trusses, frames, large networks)
- Always **verify** NetworkX assembly against hand calculations for small cases!
""")

st.divider()

# --- Interactive Demo ---
st.header("7. Try It Yourself")

n_nodes = st.slider("Number of Masses", 3, 20, 5)
graph_type = st.selectbox("Graph Topology", ["Chain", "Ring", "Random", "Star"])

# Create graph
if graph_type == "Chain":
    G = nx.path_graph(n_nodes)
elif graph_type == "Ring":
    G = nx.cycle_graph(n_nodes)
elif graph_type == "Random":
    G = nx.erdos_renyi_graph(n_nodes, 0.3)
else:  # Star
    G = nx.star_graph(n_nodes - 1)

# Show graph
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Graph Structure")
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', 
            node_size=400, ax=ax, edge_color='gray')
    st.pyplot(fig)

with col2:
    st.subheader("Stiffness Matrix")
    K = nx.laplacian_matrix(G).toarray() + np.eye(n_nodes) * 0.1
    st.dataframe(np.round(K, 2), width='stretch')

# Compute eigenvalues
eigenvalues = np.linalg.eigvalsh(K)
frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

st.subheader("Natural Frequencies")
st.write(f"Found {len(frequencies)} modes")
st.write(f"Lowest: {frequencies[0]:.3f} Hz")
st.write(f"Highest: {frequencies[-1]:.3f} Hz")

# Histogram
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(frequencies, bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Count')
ax.set_title('Frequency Distribution')
ax.grid(True, alpha=0.3)
st.pyplot(fig)