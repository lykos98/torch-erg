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
            A_gen = gibbs_ministeps(A, model, feats.to(device), device, mini_steps=gibbs_steps)
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

def show_graph_grid(graphs, rows=2, cols=4, layout="spring", titles=None,outdir="graphs_out", figsize=(12,6)):
    """
    Display a grid of graphs.

    Args:
        graphs: list of networkx.Graph
        rows, cols: grid size
        layout: 'spring', 'kamada', 'circular', 'random'
        titles: optional list of titles for each subplot
        figsize: figure size
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=figsize)
    for i, G in enumerate(graphs[:rows*cols]):
        plt.subplot(rows, cols, i+1)
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
            node_size=200,
            node_color="skyblue",
            edge_color="gray",
            with_labels=False
        )
        if titles and i < len(titles):
            plt.title(titles[i], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'grid_example.png'), bbox_inches="tight")
    plt.show()


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