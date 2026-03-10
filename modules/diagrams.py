# lib/diagrams.py

def get_double_pendulum_schema():
    """Generates a Graphviz schematic of the double pendulum system."""
    return """
    digraph DoublePendulum {
        rankdir=TB;
        node [shape=circle, style=filled, fillcolor=lightblue];
        edge [color=gray40, penwidth=2];

        Pivot [shape=point, width=0.5, label=""];
        M1 [label="m1"];
        M2 [label="m2"];

        Pivot -> M1 [label="L1", taillabel="θ1"];
        M1 -> M2 [label="L2", taillabel="θ2"];
        
        subgraph cluster_legend {
            label="Legend";
            style=dashed;
            color=gray;
            L1 [shape=plaintext, label="Rod 1"];
            L2 [shape=plaintext, label="Rod 2"];
        }
    }
    """

def get_euler_lagrange_flowchart():
    """Generates a Graphviz flowchart for the E-L Derivation process."""
    return """
    digraph EL_Process {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica"];
        edge [fontname="Helvetica", fontsize=10];

        Start [label="1. Generalized\nCoordinates (q)", fillcolor="#e1f5fe"];
        Energies [label="2. Kinetic (T)\n& Potential (V)", fillcolor="#fff9c4"];
        Lagrangian [label="3. L = T - V", fillcolor="#e8f5e9"];
        EL_Eq [label="4. Apply Euler-Lagrange\n(d/dt)(∂L/∂q̇) - (∂L/∂q) = 0", fillcolor="#fce4ec"];
        EOM [label="5. Equations\nof Motion", fillcolor="#f3e5f5"];

        Start -> Energies;
        Energies -> Lagrangian;
        Lagrangian -> EL_Eq;
        EL_Eq -> EOM;
    }
    """

# Add to lib/diagrams.py

def get_lagrange_multiplier_schema():
    """Flowchart showing the Lagrange Multiplier optimization process."""
    return """
    digraph LagrangeMultipliers {
        rankdir=TB;
        node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica"];
        edge [fontname="Helvetica", fontsize=10];

        Objective [label="1. Objective Function\nf(x,y)", fillcolor="#e3f2fd"];
        Constraint [label="2. Constraint\ng(x,y) = 0", fillcolor="#fff3e0"];
        Lagrangian [label="3. Form Lagrangian\nL = f - λg", fillcolor="#e8f5e9"];
        Derivatives [label="4. Set ∇L = 0\n∂L/∂x = 0, ∂L/∂y = 0, ∂L/∂λ = 0", fillcolor="#fce4ec"];
        Solve [label="5. Solve System\nFind (x, y, λ)", fillcolor="#f3e5f5"];
        Points [label="6. Constrained\nExtrema", fillcolor="#e0f7fa"];

        Objective -> Constraint;
        Constraint -> Lagrangian;
        Lagrangian -> Derivatives;
        Derivatives -> Solve;
        Solve -> Points;
    }
    """

def get_lagrange_points_diagram():
    """Schematic of the 5 Lagrange points in the Earth-Sun system."""
    return """
    digraph LagrangePoints {
        rankdir=LR;
        node [shape=circle, style=filled, fontname="Helvetica"];
        edge [style=invis];

        Sun [label="Sun (M1)", shape=doublecircle, fillcolor="#ffeb3b", fontcolor="black"];
        L3 [label="L3", fillcolor="#ffcdd2"];
        L1 [label="L1", fillcolor="#ffcdd2"];
        Earth [label="Earth (M2)", shape=doublecircle, fillcolor="#42a5f5", fontcolor="white"];
        L2 [label="L2", fillcolor="#ffcdd2"];
        L4 [label="L4", fillcolor="#c8e6c9"];
        L5 [label="L5", fillcolor="#c8e6c9"];

        Sun -> L3 -> L1 -> Earth -> L2;
        L4 -> Earth;
        L5 -> Earth;

        {rank=same; Sun; L3; L1; Earth; L2}
        {rank=same; L4}
        {rank=same; L5}
    }
    """