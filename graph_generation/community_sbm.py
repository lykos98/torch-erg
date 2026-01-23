"""
Community-based Stochastic Block Model (SBM) Graph Dataset Generator.

Generates graphs with clear community structure where:
- Nodes within the same community have high edge probability (p_in)
- Nodes across communities have low edge probability (p_out)
- Node features are one-hot encodings of community membership

Output format: List[GraphTuple] where each GraphTuple contains:
    - adj: (N, N) binary adjacency matrix (symmetric, zero diagonal)
    - node_features: (N, num_communities) one-hot community encoding
"""

import os
import json
import pickle
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch.utils.data import Dataset

from src.torch_erg.samplers import GraphTuple


def _generate_community_sizes(
    num_communities: int,
    total_nodes: int,
    rng: np.random.Generator,
    min_size: int = 2
) -> List[int]:
    """
    Generate random community sizes that sum to total_nodes.
    
    Each community gets at least min_size nodes.
    
    Args:
        num_communities: Number of communities
        total_nodes: Total number of nodes
        rng: Random number generator
        min_size: Minimum nodes per community
    
    Returns:
        List of community sizes
    """
    if num_communities * min_size > total_nodes:
        raise ValueError(
            f"Cannot have {num_communities} communities with min {min_size} nodes "
            f"each (total required: {num_communities * min_size}, available: {total_nodes})"
        )
    
    remaining_nodes = total_nodes - num_communities * min_size
    cuts = sorted(rng.random(num_communities - 1))
    
    sizes = []
    prev_cut = 0
    for cut in cuts:
        num_nodes = min_size + int(remaining_nodes * (cut - prev_cut))
        sizes.append(num_nodes)
        prev_cut = cut
    
    sizes.append(total_nodes - sum(sizes))
    
    return sizes


def _sample_sbm_edges(
    community_of_node: List[int],
    p_in: float,
    p_out: float,
    rng: np.random.Generator
) -> torch.Tensor:
    """
    Sample edges according to SBM probabilities.
    
    Args:
        community_of_node: List where index i gives community of node i
        p_in: Probability of edge within same community
        p_out: Probability of edge between different communities
        rng: Random number generator
    
    Returns:
        (N, N) binary adjacency matrix
    """
    n = len(community_of_node)
    adj = torch.zeros(n, n, dtype=torch.float32)
    
    for i in range(n):
        for j in range(i + 1, n):
            if community_of_node[i] == community_of_node[j]:
                prob = p_in
            else:
                prob = p_out
            
            if rng.random() < prob:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    
    return adj


def generate_community_sbm(
    num_communities: int,
    total_nodes: int,
    p_in: float = 0.3,
    p_out: float = 0.05,
    min_size: int = 2,
    seed: int = 42
) -> GraphTuple:
    """
    Generate a single SBM graph with community structure.
    
    Args:
        num_communities: Number of communities
        total_nodes: Total number of nodes
        p_in: Edge probability within same community (high)
        p_out: Edge probability between communities (low)
        min_size: Minimum nodes per community
        seed: Random seed for reproducibility
    
    Returns:
        GraphTuple with:
            - adj: (N, N) binary adjacency matrix
            - node_features: (N, num_communities) one-hot community encoding
    """
    rng = np.random.default_rng(seed)
    
    sizes = _generate_community_sizes(num_communities, total_nodes, rng, min_size=min_size)
    
    community_of_node = []
    for c, size in enumerate(sizes):
        community_of_node.extend([c] * size)
    
    adj = _sample_sbm_edges(community_of_node, p_in, p_out, rng)
    
    node_features = F.one_hot(
        torch.tensor(community_of_node),
        num_classes=num_communities
    ).float()
    
    adj = adj.fill_diagonal_(0.0)


    perm = np.random.permutation(adj.shape[0])
    adj = adj[perm,:]
    adj = adj[:,perm]
    node_features = node_features[perm,:]
    
    return GraphTuple(
        adj=adj,
        node_features=node_features
    )


def generate_community_dataset(
    num_graphs: int,
    num_communities: int,
    total_nodes: int,
    p_in: float = 0.3,
    p_out: float = 0.05,
    min_size: int = 2,
    seed: int = 42,
    save_dir: str = "data/community_sbm"
) -> List[GraphTuple]:
    """
    Generate a dataset of community-structured graphs.
    
    Args:
        num_graphs: Number of graphs to generate
        num_communities: Number of communities per graph
        total_nodes: Total nodes per graph
        p_in: Intra-community edge probability
        p_out: Inter-community edge probability
        min_size: Minimum nodes per community
        seed: Base random seed (each graph gets seed + i)
        save_dir: Base directory to save datasets
    
    Returns:
        List of GraphTuple objects
    """
    base_rng = np.random.default_rng(seed)
    
    graphs = []
    community_assignments = []
    
    for i in range(num_graphs):
        graph_seed = base_rng.integers(0, 2**31 - 1)
        
        graph = generate_community_sbm(
            num_communities=num_communities,
            total_nodes=total_nodes,
            p_in=p_in,
            p_out=p_out,
            min_size=min_size,
            seed=graph_seed
        )
        
        graphs.append(graph)
        
        comm_assignment = torch.argmax(graph.node_features, dim=1).numpy()
        community_assignments.append(comm_assignment)
    
    if save_dir is not None:
        _save_dataset(
            graphs=graphs,
            community_assignments=np.array(community_assignments),
            num_graphs=num_graphs,
            num_communities=num_communities,
            total_nodes=total_nodes,
            p_in=p_in,
            p_out=p_out,
            min_size=min_size,
            base_seed=seed,
            save_dir=save_dir
        )
    
    return graphs


def _save_dataset(
    graphs: List[GraphTuple],
    community_assignments: np.ndarray,
    num_graphs: int,
    num_communities: int,
    total_nodes: int,
    p_in: float,
    p_out: float,
    min_size: int,
    base_seed: int,
    save_dir: str
) -> None:
    """
    Save dataset to disk with metadata and README.
    """
    dir_name = f"params_{num_communities}_{total_nodes}_{p_in}_{p_out}_{min_size}"
    out_path = os.path.join(save_dir, dir_name)
    os.makedirs(out_path, exist_ok=True)
    
    graphs_path = os.path.join(out_path, "graphs.pkl")
    with open(graphs_path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    metadata = {
        "num_graphs": num_graphs,
        "num_communities": num_communities,
        "total_nodes": total_nodes,
        "p_in": p_in,
        "p_out": p_out,
        "min_size": min_size,
        "base_seed": int(base_seed),
        "description": "Community-based SBM graphs with one-hot community features",
        "format": {
            "graphs.pkl": "List[GraphTuple] - each GraphTuple has adj (N,N) and node_features (N,C)",
            "community_assignments.npy": "(num_graphs, total_nodes) int array of community labels",
            "metadata.json": "This file",
            "README.md": "Generation command and usage examples"
        }
    }
    
    metadata_path = os.path.join(out_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    assignments_path = os.path.join(out_path, "community_assignments.npy")
    np.save(assignments_path, community_assignments)
    
    readme_path = os.path.join(out_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(_generate_readme(
            num_graphs=num_graphs,
            num_communities=num_communities,
            total_nodes=total_nodes,
            p_in=p_in,
            p_out=p_out,
            min_size=min_size,
            base_seed=base_seed
        ))
    
    print(f"[INFO] Saved dataset to {out_path}")
    print(f"       - graphs.pkl: {len(graphs)} graphs")
    print(f"       - metadata.json: generation parameters")
    print(f"       - community_assignments.npy: community labels")
    print(f"       - README.md: usage documentation")


def _generate_readme(
    num_graphs: int,
    num_communities: int,
    total_nodes: int,
    p_in: float,
    p_out: float,
    min_size: int,
    base_seed: int
) -> str:
    """Generate README.md content for saved dataset."""
    return f"""# Community SBM Dataset

## Generation Parameters

| Parameter | Value |
|-----------|-------|
| num_graphs | {num_graphs} |
| num_communities | {num_communities} |
| total_nodes | {total_nodes} |
| p_in (intra-community) | {p_in} |
| p_out (inter-community) | {p_out} |
| min_size (min nodes per community) | {min_size} |
| base_seed | {base_seed} |

## Dataset Format

### Files

- **graphs.pkl**: List of `{num_graphs}` GraphTuple objects
- **community_assignments.npy**: Array of shape `({num_graphs}, {total_nodes})` with community labels
- **metadata.json**: Generation parameters in JSON format
- **README.md**: This file

### GraphTuple Structure

Each graph is stored as a `GraphTuple` with:
- `adj`: `torch.Tensor` of shape `({total_nodes}, {total_nodes})`
  - Binary adjacency matrix (0 or 1)
  - Symmetric: `adj[i,j] == adj[j,i]`
  - Zero diagonal (no self-loops)
  
- `node_features`: `torch.Tensor` of shape `({total_nodes}, {num_communities})`
  - One-hot encoding of community membership
  - `node_features[i, c] == 1` if node i belongs to community c

## Usage Examples

### Python

```python
import pickle
import torch

# Load dataset
with open("graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

# Access individual graph
graph = graphs[0]
adj = graph.adj              # ({total_nodes}, {total_nodes})
features = graph.node_features  # ({total_nodes}, {num_communities})

# Get community assignments
community_assignments = torch.argmax(features, dim=1)
```

### PyTorch DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class SBMGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        g = self.graphs[idx]
        return g.adj, g.node_features

dataset = SBMGraphDataset(graphs)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for adj_batch, feats_batch in loader:
    # adj_batch: (B, N, N)
    # feats_batch: (B, N, C)
    pass
```

## Generation Command

```bash
python -c "
from graph_generation import generate_community_dataset

generate_community_dataset(
    num_graphs={num_graphs},
    num_communities={num_communities},
    total_nodes={total_nodes},
    p_in={p_in},
    p_out={p_out},
    min_size={min_size},
    seed={base_seed},
    save_dir='data/community_sbm'
)
```
"""


def load_community_dataset(path: str) -> Tuple[List[GraphTuple], Dict[str, Any]]:
    """
    Load a saved community dataset.
    
    Args:
        path: Path to graphs.pkl file
    
    Returns:
        Tuple of (graphs list, metadata dict)
    """
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
    
    metadata_path = os.path.join(os.path.dirname(path), "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return graphs, metadata


class CommunitySBMDataset(Dataset):
    """
    PyTorch Dataset wrapper for community SBM graphs.
    
    Returns tuples of (adjacency, node_features) compatible with DataLoader.
    """
    
    def __init__(self, graphs: List[GraphTuple]):
        """
        Args:
            graphs: List of GraphTuple objects
        """
        self.graphs = graphs
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single graph.
        
        Returns:
            Tuple of (adj, node_features)
                adj: (N, N) binary adjacency
                node_features: (N, num_communities) one-hot
        """
        graph = self.graphs[idx]
        return graph.adj, graph.node_features


def compute_observables(graphs: List[GraphTuple]) -> Dict[str, np.ndarray]:
    """
    Compute basic observables for a list of graphs.
    
    Args:
        graphs: List of GraphTuple objects
    
    Returns:
        Dictionary with arrays of:
            - num_edges: Edge counts per graph
            - num_triangles: Triangle counts per graph
            - avg_degree: Average degree per graph
            - clustering_coeff: Average clustering coefficient per graph
    """
    num_edges = []
    num_triangles = []
    avg_degree = []
    clustering_coeff = []
    
    for graph in graphs:
        adj = graph.adj.numpy()
        n = adj.shape[0]
        
        edges = int(adj.sum() / 2)
        num_edges.append(edges)
        
        triangles = int(np.trace(adj @ adj @ adj) / 6)
        num_triangles.append(triangles)
        
        degrees = adj.sum(axis=1)
        avg_degree.append(float(degrees.mean()))
        
        G = nx.from_numpy_array(adj)
        cc = nx.average_clustering(G)
        clustering_coeff.append(cc)
    
    return {
        "num_edges": np.array(num_edges),
        "num_triangles": np.array(num_triangles),
        "avg_degree": np.array(avg_degree),
        "clustering_coeff": np.array(clustering_coeff)
    }


def print_dataset_stats(graphs: List[GraphTuple], name: str = "Dataset") -> None:
    """
    Print statistics for a dataset of graphs.
    """
    obs = compute_observables(graphs)
    
    print(f"\n{name} Statistics:")
    print("-" * 40)
    print(f"  Number of graphs: {len(graphs)}")
    print(f"  Nodes per graph: {graphs[0].adj.shape[0]}")
    print(f"  Features per node: {graphs[0].node_features.shape[1]}")
    print(f"  Edges: {obs['num_edges'].mean():.1f} ± {obs['num_edges'].std():.1f}")
    print(f"  Triangles: {obs['num_triangles'].mean():.1f} ± {obs['num_triangles'].std():.1f}")
    print(f"  Avg Degree: {obs['avg_degree'].mean():.2f} ± {obs['avg_degree'].std():.2f}")
    print(f"  Clustering Coeff: {obs['clustering_coeff'].mean():.3f} ± {obs['clustering_coeff'].std():.3f}")
