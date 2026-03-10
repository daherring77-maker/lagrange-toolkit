import streamlit as st

# ----------------------------
# Main App
# ----------------------------
pages = {
        "Lagrange Level 1": [
           st.Page("pages/1_newton.py", title="Newton's Laws", icon="⚙️"),
           st.Page("pages/2_least_action.py", title="Principle of Least Action", icon="📊"),
           st.Page("pages/3_euler_lagrange.py", title="Euler-Lagrange Equations", icon="📊"),
        ],
        "Lagrange Level 2"  : [
           st.Page("pages/6_statics.py", title="Rod in a Bowl", icon="📊"),
           st.Page("pages/8_single_pendulum.py", title="Single Pendulum", icon="📊"),
           st.Page("pages/9_double_pendulum.py", title="Double Pendulum", icon="📊"),
           st.Page("pages/10_euler_lagrange2.py", title="Euler-Lagrange", icon="📊"),
           st.Page("pages/11_lagrange_points.py", title="Lagrange Points", icon="📊"),
        ], 
         "Lagrange Level 3"  : [  
           st.Page("pages/12_state_space_mdof.py", title="State Space MDOF", icon="📊"),
           st.Page("pages/12A_networkx_explained.py", title="NetworkX Explained", icon="📊"),
           st.Page("pages/13_hamiltonian_phase_space.py", title="Hamiltonian Phase Space", icon="📊"),
           st.Page("pages/14_fea.py", title="Finite Element Analysis", icon="📊"),
           st.Page("pages/15_fea_stress.py", title="Finite Element Stress Analysis", icon="📊"),
        ]
        }
       
pg = st.navigation(pages)
pg.run()


