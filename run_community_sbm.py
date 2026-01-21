#!/usr/bin/env python3
"""
Community SBM Graph Dataset Generator and Visualization Helper.

Usage:
    python run_community_sbm.py --generate --visualize --num-graphs 500
    python run_community_sbm.py --generate-only --num-communities 4 --total-nodes 50
    python run_community_sbm.py --visualize-only --input data/community_sbm/params_3_30_0.35_0.05/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_generation import (
    generate_community_dataset,
    load_community_dataset,
    show_community_graph,
    show_community_graph_grid,
    print_dataset_stats,
    CommunitySBMDataset
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and visualize community SBM graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 500 graphs with 3 communities, 30 nodes each
    python run_community_sbm.py --generate --num-graphs 500
    
    # Generate with custom parameters
    python run_community_sbm.py --generate --num-communities 4 --total-nodes 50 --p-in 0.4
    
    # Generate and visualize
    python run_community_sbm.py --generate --visualize
    
    # Visualize existing dataset
    python run_community_sbm.py --visualize-only --input data/community_sbm/params_3_30_0.35_0.05/
        """
    )
    
    parser.add_argument("--generate", action="store_true", help="Generate graphs")
    parser.add_argument("--visualize", action="store_true", help="Visualize graphs")
    parser.add_argument("--generate-only", action="store_true", help="Only generate (no visualize)")
    parser.add_argument("--visualize-only", action="store_true", help="Only visualize existing dataset")
    
    parser.add_argument("--num-graphs", type=int, default=500, help="Number of graphs to generate")
    parser.add_argument("--num-communities", type=int, default=3, help="Number of communities")
    parser.add_argument("--total-nodes", type=int, default=30, help="Nodes per graph")
    parser.add_argument("--p-in", type=float, default=0.35, help="Intra-community edge probability")
    parser.add_argument("--p-out", type=float, default=0.05, help="Inter-community edge probability")
    parser.add_argument("--min-size", type=int, default=3, help="Minimum nodes per community")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="data/community_sbm", help="Save directory")
    
    parser.add_argument("--input", type=str, default=None, help="Input dataset path for visualization")
    parser.add_argument("--output-dir", type=str, default="results/visualization", help="Output directory for visualizations")
    parser.add_argument("--grid-rows", type=int, default=2, help="Rows in grid visualization")
    parser.add_argument("--grid-cols", type=int, default=4, help="Columns in grid visualization")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not any([args.generate, args.visualize, args.generate_only, args.visualize_only]):
        args.generate = True
        args.visualize = True
    
    if args.generate or args.generate_only:
        print("=" * 60)
        print("GENERATING COMMUNITY SBM DATASET")
        print("=" * 60)
        print(f"  num_graphs: {args.num_graphs}")
        print(f"  num_communities: {args.num_communities}")
        print(f"  total_nodes: {args.total_nodes}")
        print(f"  p_in: {args.p_in}")
        print(f"  p_out: {args.p_out}")
        print(f"  min_size: {args.min_size}")
        print(f"  seed: {args.seed}")
        print()
        
        dataset = generate_community_dataset(
            num_graphs=args.num_graphs,
            num_communities=args.num_communities,
            total_nodes=args.total_nodes,
            p_in=args.p_in,
            p_out=args.p_out,
            min_size=args.min_size,
            seed=args.seed,
            save_dir=args.save_dir
        )
        
        print()
        print_dataset_stats(dataset, "Generated Dataset")
        
        dataset_path = os.path.join(
            args.save_dir,
            f"params_{args.num_communities}_{args.total_nodes}_{args.p_in}_{args.p_out}_{args.min_size}/graphs.pkl"
        )
        input_path = args.input or dataset_path
    
    if args.visualize or args.visualize_only:
        if args.visualize_only and args.input:
            input_path = args.input
        elif args.generate or args.generate_only:
            input_path = os.path.join(
                args.save_dir,
                f"params_{args.num_communities}_{args.total_nodes}_{args.p_in}_{args.p_out}_{args.min_size}/graphs.pkl"
            )
        else:
            input_path = args.input
        
        print()
        print("=" * 60)
        print("LOADING DATASET FOR VISUALIZATION")
        print("=" * 60)
        print(f"  Input path: {input_path}")
        
        dataset, metadata = load_community_dataset(input_path)
        print(f"  Loaded {len(dataset)} graphs")
        print()
        
        print("=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"  Output directory: {args.output_dir}")
        print()
        
        print("  [1/2] Generating single graph visualization...")
        show_community_graph(
            dataset[0],
            outdir=args.output_dir,
            filename="example_graph"
        )
        
        print("  [2/2] Generating grid visualization...")
        num_show = min(args.grid_rows * args.grid_cols, len(dataset))
        show_community_graph_grid(
            dataset[:num_show],
            rows=args.grid_rows,
            cols=args.grid_cols,
            outdir=args.output_dir,
            filename="graph_grid"
        )
        
        print()
        print(f"Visualizations saved to: {args.output_dir}/")
        print(f"  - example_graph.png/pdf")
        print(f"  - graph_grid.png/pdf")
    
    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
