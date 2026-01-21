"""
Test script for community SBM graph generation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_generation import (
    generate_community_dataset,
    load_community_dataset,
    show_community_graph,
    show_community_graph_grid,
    print_dataset_stats,
    CommunitySBMDataset
)
from torch.utils.data import DataLoader


def test_single_graph():
    """Test generating a single graph."""
    print("=" * 60)
    print("TEST: Single Graph Generation")
    print("=" * 60)
    
    from graph_generation import generate_community_sbm
    
    graph = generate_community_sbm(
        num_communities=3,
        total_nodes=30,
        p_in=0.35,
        p_out=0.05,
        seed=42
    )
    
    print(f"Adjacency shape: {graph.adj.shape}")
    print(f"Node features shape: {graph.node_features.shape}")
    print(f"Is symmetric: {(graph.adj == graph.adj.T).all().item()}")
    print(f"Diagonal is zero: {graph.adj.diagonal().sum().item() == 0}")
    print(f"Feature sum per node: {graph.node_features.sum(dim=1).tolist()[:5]}...")
    
    communities = graph.node_features.argmax(dim=1)
    comm_counts = communities.bincount().tolist()
    print(f"Community sizes: {comm_counts}")
    
    return graph


def test_dataset_generation_and_save():
    """Test generating and saving a dataset."""
    print("\n" + "=" * 60)
    print("TEST: Dataset Generation and Save")
    print("=" * 60)
    
    dataset = generate_community_dataset(
        num_graphs=20,
        num_communities=3,
        total_nodes=30,
        p_in=0.35,
        p_out=0.05,
        seed=42,
        save_dir="data/community_sbm"
    )
    
    print(f"Generated {len(dataset)} graphs")
    print(f"First graph adj shape: {dataset[0].adj.shape}")
    print(f"First graph features shape: {dataset[0].node_features.shape}")
    
    print_dataset_stats(dataset, "Generated Dataset")
    
    return dataset


def test_load_dataset():
    """Test loading a saved dataset."""
    print("\n" + "=" * 60)
    print("TEST: Dataset Loading")
    print("=" * 60)
    
    graphs, metadata = load_community_dataset(
        "data/community_sbm/params_3_30_0.35_0.05/graphs.pkl"
    )
    
    print(f"Loaded {len(graphs)} graphs")
    print(f"Metadata: {metadata}")
    
    return graphs, metadata


def test_visualization():
    """Test visualization functions."""
    print("\n" + "=" * 60)
    print("TEST: Visualization")
    print("=" * 60)
    
    dataset = generate_community_dataset(
        num_graphs=8,
        num_communities=3,
        total_nodes=30,
        p_in=0.35,
        p_out=0.05,
        seed=123,
        save_dir=None
    )
    
    os.makedirs("results/visualization_test", exist_ok=True)
    
    show_community_graph(
        dataset[0],
        outdir="results/visualization_test",
        filename="single_graph"
    )
    
    show_community_graph_grid(
        dataset[:8],
        rows=2,
        cols=4,
        outdir="results/visualization_test",
        filename="graph_grid"
    )
    
    print("Visualizations saved to results/visualization_test/")


def test_pytorch_integration():
    """Test integration with PyTorch DataLoader."""
    print("\n" + "=" * 60)
    print("TEST: PyTorch DataLoader Integration")
    print("=" * 60)
    
    dataset = generate_community_dataset(
        num_graphs=32,
        num_communities=4,
        total_nodes=25,
        p_in=0.4,
        p_out=0.03,
        seed=42,
        save_dir=None
    )
    
    torch_dataset = CommunitySBMDataset(dataset)
    loader = DataLoader(torch_dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (adj_batch, feats_batch) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  adj_batch shape: {adj_batch.shape}")  # (B, N, N)
        print(f"  feats_batch shape: {feats_batch.shape}")  # (B, N, C)
        
        if batch_idx >= 2:
            break
    
    print("PyTorch integration test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMMUNITY SBM GRAPH GENERATION - TEST SUITE")
    print("=" * 60)
    
    test_single_graph()
    test_dataset_generation_and_save()
    test_load_dataset()
    test_visualization()
    test_pytorch_integration()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
