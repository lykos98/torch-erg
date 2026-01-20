import torch
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np



from deep_ebm.gnn_ebm import gibbs_ministeps, evaluate_mmd_generated

def evaluate_model(model, dataset, device, num_graphs=50, gibbs_steps=200):
    """
    Generate graphs with the trained model and evaluate them against dataset graphs.
    
    Args:
        model: trained GNN_EBM
        dataset: GraphDataset
        device: torch.device
        num_graphs: number of graphs to generate for evaluation
        gibbs_steps: number of Gibbs mini-steps per graph
    
    Returns:
        dict with MMD statistics
    """
    model.eval()
    generated = []
    with torch.no_grad():
        for i in range(min(num_graphs, len(dataset))):
            A, feats = dataset[i]
            # run Gibbs sampling refinement
            A_gen = gibbs_ministeps(
                A,
                feats.to(device),
                model,
                device,
                mini_steps=gibbs_steps,
            )
            G_gen = nx.from_numpy_array(A_gen.cpu().numpy())
            generated.append(G_gen)

    # compare distributions of degree and clustering coefficient
    metrics = evaluate_mmd_generated(dataset.graphs[:len(generated)], generated)
    return metrics




def show_graph(G, title=None, layout="spring"):
    """
    Display a single NetworkX graph.
    """
    plt.figure(figsize=(4,4))
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)
    
    nx.draw(
        G, pos,
        node_size=300,
        node_color="skyblue",
        edge_color="gray",
        with_labels=False
    )
    if title:
        plt.title(title)
    plt.show()


def compare_graphs(G_real, G_generated, layout="spring", outdir="graphs_out"):
    """
    Show a real vs. generated graph side by side.
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,4))
    
    if layout == "spring":
        pos_real = nx.spring_layout(G_real, seed=42)
        pos_gen = nx.spring_layout(G_generated, seed=42)
    else:
        pos_real = nx.circular_layout(G_real)
        pos_gen = nx.circular_layout(G_generated)
    
    plt.subplot(1,2,1)
    nx.draw(G_real, pos_real, node_size=300, node_color="lightgreen", edge_color="gray")
    plt.title("Real graph")

    plt.subplot(1,2,2)
    nx.draw(G_generated, pos_gen, node_size=300, node_color="salmon", edge_color="gray")
    plt.title("Generated graph")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'compare_graphs.png'), bbox_inches="tight")
    plt.show()


def save_graph(G, filename, layout="spring", outdir="graphs_out"):
    """
    Save a single graph as PNG.
    """
    os.makedirs(outdir, exist_ok=True)
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.circular_layout(G)
    
    plt.figure(figsize=(4,4))
    nx.draw(G, pos, node_size=300, node_color="skyblue", edge_color="gray")
    plt.savefig(os.path.join(outdir, filename), bbox_inches="tight")
    plt.close()

from matplotlib import cm
from matplotlib.colors import Normalize


def show_graph_grid(
    graphs,
    rows=2,
    cols=4,
    layout="spring",
    titles=None,
    outdir="graphs_out",
    filename="graph_grid",
    figsize=(12, 6),
    node_size=120,
    node_color="#1f77b4",
    edge_alpha=0.35,
    edge_width=0.8,
    seed=42,
):
    """
    Publication-quality grid visualization for graphs.

    - Uniform node size and color
    - Per-graph layout (community structure visible)
    - Properly spaced suptitle (never overlaps plots)
    - Serif typography suitable for papers/theses
    """

    os.makedirs(outdir, exist_ok=True)

    # -------------------------------------------------
    # Typography (paper-grade)
    # -------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 14,
    })

    num_graphs = min(len(graphs), rows * cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        constrained_layout=False
    )
    axes = np.atleast_1d(axes).flatten()

    # -------------------------------------------------
    # Draw graphs
    # -------------------------------------------------
    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= num_graphs:
            continue

        G = graphs[i]

        # ---- per-graph layout ----
        if layout == "spring":
            pos = nx.spring_layout(
                G,
                seed=seed,
                k=1.5 / np.sqrt(G.number_of_nodes()),
                iterations=200,
            )
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # ---- edges ----
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_width,
            alpha=edge_alpha,
            edge_color="black",
        )

        # ---- nodes (uniform styling) ----
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            linewidths=0.4,
            edgecolors="black",
        )

        # ---- subplot title ----
        if titles and i < len(titles):
            ax.set_title(titles[i], pad=4)

    # -------------------------------------------------
    # Suptitle (properly reserved space)
    # -------------------------------------------------
    fig.suptitle(
        f"Sampled graphs with method {filename}",
        fontweight="semibold",
        y=0.97,
    )

    # -------------------------------------------------
    # Layout: reserve space for title
    # -------------------------------------------------
    fig.tight_layout(
        pad=0.6,
        rect=[0, 0, 1, 0.94],  # top space for suptitle
    )

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------
    png_path = os.path.join(outdir, f"{filename}.png")
    pdf_path = os.path.join(outdir, f"{filename}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()
    plt.close()



def show_graph_grid_ba(
    graphs,
    rows=2,
    cols=4,
    layout="spring",
    titles=None,
    outdir="graphs_out",
    filename="graph_grid",
    figsize=(12, 6),
    node_size=80,
    node_color="#1f77b4",
    edge_alpha=0.35,
    edge_width=0.8,
    seed=42,
    degree_scale=True,
    fixed_pos_graph=None,
):
    """
    BA-optimized grid visualization.

    - Optional degree-scaled node sizes (recommended for BA)
    - Optional fixed layout for before/after comparisons
    - Paper-quality typography and spacing
    """

    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 14,
    })

    num_graphs = min(len(graphs), rows * cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    # -------------------------------------------------
    # Fixed layout (if provided)
    # -------------------------------------------------
    fixed_pos = None
    if fixed_pos_graph is not None:
        if layout == "spring":
            fixed_pos = nx.spring_layout(
                fixed_pos_graph,
                seed=seed,
                k=1.5 / np.sqrt(fixed_pos_graph.number_of_nodes()),
                iterations=300,
            )
        elif layout == "kamada":
            fixed_pos = nx.kamada_kawai_layout(fixed_pos_graph)

    # -------------------------------------------------
    # Draw graphs
    # -------------------------------------------------
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= num_graphs:
            continue

        G = graphs[i]

        # ---- layout ----
        if fixed_pos is not None:
            pos = fixed_pos
        else:
            if layout == "spring":
                pos = nx.spring_layout(
                    G,
                    seed=seed,
                    k=1.5 / np.sqrt(G.number_of_nodes()),
                    iterations=200,
                )
            elif layout == "kamada":
                pos = nx.kamada_kawai_layout(G)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            else:
                raise ValueError(f"Unknown layout: {layout}")

        # ---- node sizes ----
        if degree_scale:
            deg = np.array([d for _, d in G.degree()])
            sizes = node_size * (deg / (deg.mean() + 1e-6) + 0.3)
        else:
            sizes = node_size

        # ---- edges ----
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_width,
            alpha=edge_alpha,
            edge_color="black",
        )

        # ---- nodes ----
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=sizes,
            node_color=node_color,
            linewidths=0.4,
            edgecolors="black",
        )

        if titles and i < len(titles):
            ax.set_title(titles[i], pad=4)

    fig.suptitle(
        f"Sampled Barabási–Albert graphs ({filename})",
        fontweight="semibold",
        y=0.97,
    )

    fig.tight_layout(pad=0.6, rect=[0, 0, 1, 0.94])

    fig.savefig(os.path.join(outdir, f"{filename}.png"), dpi=300)
    fig.savefig(os.path.join(outdir, f"{filename}.pdf"))
    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def show_graph_grid_sbm_2comm(
    graphs,
    n_per_comm=None,
    rows=2,
    cols=4,
    layout="spring",
    titles=None,
    outdir="graphs_out",
    filename="graph_grid_2comm",
    figsize=(12, 6),
    node_size=120,
    edge_alpha=0.35,
    edge_width=0.8,
    seed=42,
):
    """
    Publication-quality grid visualization for 2-community SBM graphs.

    - Nodes are colored by ground-truth community (first half / second half)
    - Spring layout with separated initialization to preserve communities
    - Visualization-only use of labels (does NOT affect permutation invariance)
    """

    os.makedirs(outdir, exist_ok=True)

    # -------------------------------------------------
    # Typography (paper-grade)
    # -------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 14,
    })

    num_graphs = min(len(graphs), rows * cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        constrained_layout=False
    )
    axes = np.atleast_1d(axes).flatten()

    # -------------------------------------------------
    # Draw graphs
    # -------------------------------------------------
    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= num_graphs:
            continue

        G = graphs[i]
        N = G.number_of_nodes()

        # infer community size if not provided
        if n_per_comm is None:
            assert N % 2 == 0, "N must be even for equal-size SBM visualization"
            n1 = N // 2
        else:
            n1 = n_per_comm

        # ---- node colors (ground-truth blocks) ----
        node_colors = [
            "tab:blue" if v < n1 else "tab:orange"
            for v in G.nodes()
        ]

        # ---- separated initialization for spring layout ----
        if layout == "spring":
            init_pos = {}
            rng = np.random.default_rng(seed + i)

            for v in G.nodes():
                if v < n1:
                    init_pos[v] = np.array([-1.0, rng.normal(scale=0.15)])
                else:
                    init_pos[v] = np.array([+1.0, rng.normal(scale=0.15)])

            pos = nx.spring_layout(
                G,
                seed=seed,
                pos=init_pos,
                iterations=200,
            )

        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(G)

        elif layout == "circular":
            pos = nx.circular_layout(G)

        else:
            raise ValueError(f"Unknown layout: {layout}")

        # ---- edges ----
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_width,
            alpha=edge_alpha,
            edge_color="black",
        )

        # ---- nodes ----
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=node_colors,
            linewidths=0.4,
            edgecolors="black",
        )

        # ---- subplot title ----
        if titles and i < len(titles):
            ax.set_title(titles[i], pad=4)

    # -------------------------------------------------
    # Suptitle
    # -------------------------------------------------
    fig.suptitle(
        "Two-community SBM samples",
        fontweight="semibold",
        y=0.97,
    )

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    fig.tight_layout(
        pad=0.6,
        rect=[0, 0, 1, 0.94],
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    png_path = os.path.join(outdir, f"{filename}.png")
    pdf_path = os.path.join(outdir, f"{filename}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()
    plt.close()



def compute_graph_statistics(graphs):
    """
    Compute average statistics over a list of networkx graphs.
    
    Returns:
        dict with average number of nodes, edges, degree, clustering coefficient.
    """
    n_nodes = [G.number_of_nodes() for G in graphs]
    n_edges = [G.number_of_edges() for G in graphs]
    avg_degree = [np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0 for G in graphs]
    avg_clustering = [nx.average_clustering(G) if G.number_of_nodes() > 0 else 0 for G in graphs]

    stats = {
        "num_graphs": len(graphs),
        "avg_nodes": np.mean(n_nodes),
        "avg_edges": np.mean(n_edges),
        "avg_degree": np.mean(avg_degree),
        "avg_clustering": np.mean(avg_clustering),
    }
    return stats


def compare_statistics(real_graphs, generated_graphs, printout=True):
    """
    Compare dataset vs. generated graph statistics.
    """
    real_stats = compute_graph_statistics(real_graphs)
    gen_stats = compute_graph_statistics(generated_graphs)

    if printout:
        print("=== Graph Statistics Comparison ===")
        print(f"Real graphs: {real_stats['num_graphs']} graphs")
        print(f"  Avg nodes:      {real_stats['avg_nodes']:.2f}")
        print(f"  Avg edges:      {real_stats['avg_edges']:.2f}")
        print(f"  Avg degree:     {real_stats['avg_degree']:.2f}")
        print(f"  Avg clustering: {real_stats['avg_clustering']:.3f}")
        print(f"Generated graphs: {gen_stats['num_graphs']} graphs")
        print(f"  Avg nodes:      {gen_stats['avg_nodes']:.2f}")
        print(f"  Avg edges:      {gen_stats['avg_edges']:.2f}")
        print(f"  Avg degree:     {gen_stats['avg_degree']:.2f}")
        print(f"  Avg clustering: {gen_stats['avg_clustering']:.3f}")
    
    return real_stats, gen_stats



def plot_parameter_evolution(
    parlist_np: np.ndarray,
    outdir: str,
    filename: str = "parameter_evolution.png",
    title: str = "Parameter Evolution",
    burnin: int | None = None,
    dpi: int = 200,
):
    """
    Plot and save the evolution of ERGM / hybrid parameters.

    Args:
        parlist_np: np.ndarray of shape (T, P)
            Parameter values over time.
        outdir: str
            Directory where the plot is saved.
        filename: str
            Output image name.
        burnin: int or None
            Optional burn-in index to discard initial samples.
        dpi: int
            Image resolution.
    """

    os.makedirs(outdir, exist_ok=True)

    if burnin is not None:
        parlist_np = parlist_np[burnin:]

    T, P = parlist_np.shape
    print("Final parameters:", parlist_np[-1])
    print("Trajectory shape:", parlist_np.shape)

    # ---------------------------
    # Style (clean & readable)
    # ---------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
    })

    # ---------------------------
    # Figure
    # ---------------------------
    fig, axes = plt.subplots(
        1, P,
        figsize=(3.5 * P, 4.5),
        sharex=True,
        dpi=dpi
    )

    if P == 1:
        axes = [axes]

    for p in range(P):
        ax = axes[p]
        ax.plot(parlist_np[:, p], marker="o", alpha=0.85)
        ax.set_title(rf"$\beta_{{{p}}}$")
        ax.set_xlabel("Update step")
        ax.grid(True, linestyle="--", alpha=0.4)

        if p == 0:
            ax.set_ylabel("Parameter value")

    fig.suptitle(title, y=1.05)
    fig.tight_layout()

    # ---------------------------
    # Save
    # ---------------------------
    save_path = os.path.join(outdir, filename)
    
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved parameter evolution plot to: {save_path}")