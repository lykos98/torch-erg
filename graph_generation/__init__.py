"""
Community-based Stochastic Block Model Graph Generation.

This package provides utilities for generating graphs with clear community
structure and visualizing them with community-colored nodes.

Main Features:
- Generate SBM graphs with configurable community parameters
- One-hot encoded node features for community membership
- Save/load datasets with metadata
- Visualization with Tab10 colormap and Kamada-Kawai layout
- Compatible with PyTorch DataLoader
"""

from .community_sbm import (
    generate_community_sbm,
    generate_community_dataset,
    load_community_dataset,
    CommunitySBMDataset,
    compute_observables,
    print_dataset_stats,
)

from .visualization import (
    show_community_graph,
    show_community_graph_grid,
    show_community_comparison,
    plot_observables_distribution,
    TAB10_COLORS,
)

from src.torch_erg.samplers import GraphTuple

__all__ = [
    "generate_community_sbm",
    "generate_community_dataset",
    "load_community_dataset",
    "CommunitySBMDataset",
    "compute_observables",
    "print_dataset_stats",
    "show_community_graph",
    "show_community_graph_grid",
    "show_community_comparison",
    "plot_observables_distribution",
    "TAB10_COLORS",
    "GraphTuple",
]

__version__ = "0.1.0"
