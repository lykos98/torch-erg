"""
Visualization utilities for community-structured graphs.

Provides functions to visualize graphs with nodes colored by community
using Tab10 colormap and Kamada-Kawai layout.
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx
import torch

from src.torch_erg.samplers import GraphTuple


TAB10_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


def _get_community_colors(node_features: torch.Tensor) -> List[str]:
    """
    Get color list for each node based on community membership.
    
    Args:
        node_features: (N, C) one-hot tensor
    
    Returns:
        List of color strings for each node
    """
    communities = torch.argmax(node_features, dim=1).numpy()
    colors = [TAB10_COLORS[c % len(TAB10_COLORS)] for c in communities]
    return colors


def _graph_tuple_to_networkx(graph: GraphTuple) -> nx.Graph:
    """
    Convert GraphTuple to NetworkX graph.
    
    Args:
        graph: GraphTuple with adj matrix
    
    Returns:
        NetworkX Graph
    """
    adj = graph.adj.numpy()
    G = nx.from_numpy_array(adj)
    return G


def show_community_graph(
    graph: GraphTuple,
    layout: str = "kamada",
    figsize: Tuple[int, int] = (6, 6),
    node_size: int = 200,
    edge_width: float = 1.0,
    edge_alpha: float = 0.5,
    outdir: Optional[str] = None,
    filename: str = "community_graph",
    seed: int = 42
) -> Optional[plt.Figure]:
    """
    Visualize a single community-structured graph.
    
    Args:
        graph: GraphTuple to visualize
        layout: Layout algorithm ("kamada", "spring", "circular")
        figsize: Figure size (width, height) in inches
        node_size: Size of nodes
        edge_width: Width of edges
        edge_alpha: Transparency of edges
        outdir: Output directory (if None, just display)
        filename: Base filename for output
        seed: Random seed for layout
    
    Returns:
        matplotlib Figure if outdir is None, else None
    """
    G = _graph_tuple_to_networkx(graph)
    colors = _get_community_colors(graph.node_features)
    
    plt.rcParams.update({
        #"font.family": "serif",
        #"font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 14,
        "figure.titlesize": 16,
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.axis("off")
    
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(G.number_of_nodes()))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_width,
        alpha=edge_alpha,
        edge_color="gray"
    )
    
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=colors,
        linewidths=0.5,
        edgecolors="black"
    )
    
    num_communities = graph.node_features.shape[1]
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=TAB10_COLORS[i], markersize=10, label=f'Community {i}')
        for i in range(min(num_communities, len(TAB10_COLORS)))
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    ax.set_title(f"Community SBM Graph\n({G.number_of_nodes()} nodes, {num_communities} communities)", 
                 fontweight="semibold")
    
    fig.tight_layout()
    
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        png_path = os.path.join(outdir, f"{filename}.png")
        pdf_path = os.path.join(outdir, f"{filename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved visualization to {png_path}")
    else:
        plt.show()
        return fig


def show_community_graph_grid(
    graphs: List[GraphTuple],
    rows: int = 2,
    cols: int = 4,
    layout: str = "kamada",
    figsize: Optional[Tuple[int, int]] = None,
    node_size: int = 120,
    edge_width: float = 0.8,
    edge_alpha: float = 0.4,
    outdir: Optional[str] = None,
    filename: str = "community_grid",
    seed: int = 42
) -> Optional[plt.Figure]:
    """
    Visualize a grid of community-structured graphs.
    
    Args:
        graphs: List of GraphTuple objects to visualize
        rows: Number of rows in grid
        cols: Number of columns in grid
        layout: Layout algorithm ("kamada", "spring", "circular")
        figsize: Custom figure size (if None, auto-calculated)
        node_size: Size of nodes
        edge_width: Width of edges
        edge_alpha: Transparency of edges
        outdir: Output directory (if None, just display)
        filename: Base filename for output
        seed: Random seed for layout
    
    Returns:
        matplotlib Figure if outdir is None, else None
    """
    num_graphs = min(len(graphs), rows * cols)
    
    if figsize is None:
        figsize = (4 * cols, 4 * rows)
    
    plt.rcParams.update({
        #"font.family": "serif",
        #"font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 14,
    })
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    num_communities = graphs[0].node_features.shape[1]
    
    for i, ax in enumerate(axes):
        ax.axis("off")
        
        if i >= num_graphs:
            continue
        
        graph = graphs[i]
        G = _graph_tuple_to_networkx(graph)
        colors = _get_community_colors(graph.node_features)
        
        if layout == "kamada":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, seed=seed + i, k=1.5 / np.sqrt(G.number_of_nodes()))
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_width,
            alpha=edge_alpha,
            edge_color="black"
        )
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_size,
            node_color=colors,
            linewidths=0.4,
            edgecolors="black"
        )
        
        edge_count = G.number_of_edges()
        ax.set_title(f"Graph {i+1}\n({edge_count} edges)", pad=2)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=TAB10_COLORS[c], markersize=8, label=f'C{c}')
        for c in range(min(num_communities, len(TAB10_COLORS)))
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(num_communities, 6),
        fontsize=9
    )
    
    fig.suptitle(
        f"Community SBM Graphs ({num_communities} communities per graph)",
        fontweight="semibold",
        y=0.98
    )
    
    fig.tight_layout(
        pad=0.6,
        rect=[0, 0.04, 1, 0.94]
    )
    
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        png_path = os.path.join(outdir, f"{filename}.png")
        pdf_path = os.path.join(outdir, f"{filename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved grid visualization to {png_path}")
    else:
        plt.show()
        return fig


def show_community_comparison(
    graphs_real: List[GraphTuple],
    graphs_generated: List[GraphTuple],
    rows: int = 2,
    cols: int = 4,
    layout: str = "kamada",
    figsize: Tuple[int, int] = (12, 6),
    node_size: int = 100,
    edge_width: float = 0.6,
    outdir: Optional[str] = None,
    filename: str = "community_comparison",
    seed: int = 42
) -> Optional[plt.Figure]:
    """
    Compare real and generated community graphs side by side.
    
    Args:
        graphs_real: List of real graphs
        graphs_generated: List of generated graphs
        rows: Rows per section (real/generated)
        cols: Columns
        layout: Layout algorithm
        figsize: Figure size
        node_size: Node size
        edge_width: Edge width
        outdir: Output directory
        filename: Base filename
        seed: Random seed
    """
    plt.rcParams.update({
        #"font.family": "serif",
        #"font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 14,
    })
    
    num_show = min(rows * cols // 2, len(graphs_real), len(graphs_generated))
    
    fig, axes = plt.subplots(2, cols, figsize=figsize)
    axes = axes.flatten()
    
    num_communities = graphs_real[0].node_features.shape[1]
    
    for i, ax in enumerate(axes):
        ax.axis("off")
    
    for i in range(num_show):
        real_graph = graphs_real[i]
        gen_graph = graphs_generated[i]
        
        G_real = _graph_tuple_to_networkx(real_graph)
        G_gen = _graph_tuple_to_networkx(gen_graph)
        
        colors_real = _get_community_colors(real_graph.node_features)
        colors_gen = _get_community_colors(gen_graph.node_features)
        
        if layout == "kamada":
            pos_real = nx.kamada_kawai_layout(G_real)
            pos_gen = nx.kamada_kawai_layout(G_gen)
        elif layout == "spring":
            pos_real = nx.spring_layout(G_real, seed=seed + i, k=1.5 / np.sqrt(G_real.number_of_nodes()))
            pos_gen = nx.spring_layout(G_gen, seed=seed + i + 1000, k=1.5 / np.sqrt(G_gen.number_of_nodes()))
        else:
            pos_real = nx.circular_layout(G_real)
            pos_gen = nx.circular_layout(G_gen)
        
        nx.draw_networkx_edges(G_real, pos_real, ax=axes[i], width=edge_width, alpha=0.4, edge_color="gray")
        nx.draw_networkx_nodes(G_real, pos_real, ax=axes[i], node_size=node_size, node_color=colors_real, linewidths=0.3, edgecolors="black")
        axes[i].set_title(f"Real {i+1}", pad=2)
        
        idx = i + cols
        nx.draw_networkx_edges(G_gen, pos_gen, ax=axes[idx], width=edge_width, alpha=0.4, edge_color="gray")
        nx.draw_networkx_nodes(G_gen, pos_gen, ax=axes[idx], node_size=node_size, node_color=colors_gen, linewidths=0.3, edgecolors="black")
        axes[idx].set_title(f"Generated {i+1}", pad=2)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=TAB10_COLORS[c], markersize=8, label=f'C{c}')
        for c in range(min(num_communities, len(TAB10_COLORS)))
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(num_communities, 6),
        fontsize=9
    )
    
    fig.suptitle("Real vs Generated Community SBM Graphs", fontweight="semibold", y=0.98)
    
    fig.tight_layout(pad=0.6, rect=[0, 0.03, 1, 0.94])
    
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        png_path = os.path.join(outdir, f"{filename}.png")
        pdf_path = os.path.join(outdir, f"{filename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved comparison visualization to {png_path}")
    else:
        plt.show()
        return fig


def plot_observables_distribution(
    graphs: List[GraphTuple],
    outdir: Optional[str] = None,
    filename: str = "observables_dist"
) -> Optional[plt.Figure]:
    """
    Plot distribution of graph observables.
    
    Args:
        graphs: List of GraphTuple objects
        outdir: Output directory
        filename: Base filename
    """
    from graph_generation.community_sbm import compute_observables
    
    obs = compute_observables(graphs)
    
    plt.rcParams.update({
        #"font.family": "serif",
        #"font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].hist(obs["num_edges"], bins=20, edgecolor='black', alpha=0.7, color=TAB10_COLORS[0])
    axes[0, 0].set_xlabel("Number of Edges")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Edge Count Distribution\n(mean={obs['num_edges'].mean():.1f}, std={obs['num_edges'].std():.1f})")
    
    axes[0, 1].hist(obs["num_triangles"], bins=20, edgecolor='black', alpha=0.7, color=TAB10_COLORS[1])
    axes[0, 1].set_xlabel("Number of Triangles")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Triangle Count Distribution\n(mean={obs['num_triangles'].mean():.1f}, std={obs['num_triangles'].std():.1f})")
    
    axes[1, 0].hist(obs["avg_degree"], bins=20, edgecolor='black', alpha=0.7, color=TAB10_COLORS[2])
    axes[1, 0].set_xlabel("Average Degree")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"Average Degree Distribution\n(mean={obs['avg_degree'].mean():.2f}, std={obs['avg_degree'].std():.2f})")
    
    axes[1, 1].hist(obs["clustering_coeff"], bins=20, edgecolor='black', alpha=0.7, color=TAB10_COLORS[3])
    axes[1, 1].set_xlabel("Clustering Coefficient")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"Clustering Coefficient Distribution\n(mean={obs['clustering_coeff'].mean():.3f}, std={obs['clustering_coeff'].std():.3f})")
    
    fig.suptitle("Community SBM Graph Observables", fontweight="semibold", y=1.02)
    fig.tight_layout()
    
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        png_path = os.path.join(outdir, f"{filename}.png")
        pdf_path = os.path.join(outdir, f"{filename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved observables distribution to {png_path}")
    else:
        plt.show()
        return fig
