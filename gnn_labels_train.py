import os
import time
import math
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from graph_generation.visualization import show_community_graph, show_community_graph_grid
from graph_generation.community_sbm import load_community_dataset, GraphTuple, CommunitySBMDataset 

# -----------------------------
# Your utils
# -----------------------------
from deep_ebm.labels_sampling import (
    train_pcd_batched_joint,
    blocked_gibbs_ministeps_batch,
    graph_from_adj_labels,
)

# -----------------------------
# Your TGNN
# -----------------------------
from deep_ebm.stable_gnn import TGNN_EBM


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Plotting: save grid (works with your plot_graph)
# ============================================================

def save_graph_grid(
    A_list,
    y_list,
    out_path: str,
    rows: int,
    cols: int,
    layout: str = "spring",
    seed: int = 0,
    titles=None,
    figsize=None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if figsize is None:
        figsize = (4 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for k, ax in enumerate(axes):
        if k >= len(A_list):
            ax.axis("off")
            continue

        A = A_list[k]
        y = y_list[k]
        G = graph_from_adj_labels(A, y)

        title = None
        if titles is not None and k < len(titles):
            title = titles[k]

        plot_graph(
            G,
            labels=y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y),
            layout=layout,
            seed=seed + k,
            ax=ax,
            title=title,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# Sampling from the trained joint EBM (long mixing, good results)
# ============================================================

@torch.no_grad()
def sample_from_ebm_joint(
    model,
    device,
    N: int,
    num_classes: int = 2,
    n_chains: int = 16,
    burnin: int = 6000,
    thin: int = 400,
    n_collect: int = 16,
    init_edge_p: float = 0.1,
    p_edge: float = 0.8,
):
    """
    Samples (A, labels) from the joint EBM using blocked Gibbs.

    Design choices for mixing quality:
      - Use many burn-in steps
      - Then collect with thinning to reduce autocorrelation
      - Keep p_edge high (edge updates) after training so structure refines
        while labels still occasionally move

    Returns:
      A_samples: list of [N,N] tensors on CPU
      y_samples: list of [N] tensors on CPU
    """

    model.eval()

    # --- init chains ---
    A0 = torch.bernoulli(torch.full((n_chains, N, N), init_edge_p, device=device))
    A0 = torch.triu(A0, diagonal=1)
    A0 = A0 + A0.transpose(1, 2)

    # random labels
    y0 = torch.randint(0, num_classes, (n_chains, N), device=device)

    A = A0
    y = y0

    # --- burn-in ---
    # we call your blocked sampler in chunks to avoid giant Python loops overhead
    # each call runs `mini_steps` blocked updates
    chunk = 200
    steps_done = 0
    while steps_done < burnin:
        run = min(chunk, burnin - steps_done)
        A, y = blocked_gibbs_ministeps_batch(
            A, y, model, device, mini_steps=run, num_classes=num_classes, p_edge=p_edge
        )
        steps_done += run

    # --- collect with thinning ---
    A_samples, y_samples = [], []
    collected = 0

    while collected < n_collect:
        # thin
        steps_done = 0
        while steps_done < thin:
            run = min(chunk, thin - steps_done)
            A, y = blocked_gibbs_ministeps_batch(
                A, y, model, device, mini_steps=run, num_classes=num_classes, p_edge=p_edge
            )
            steps_done += run

        # snapshot all chains, or just take as many as needed
        for b in range(n_chains):
            if collected >= n_collect:
                break
            A_samples.append(A[b].detach().cpu())
            y_samples.append(y[b].detach().cpu())
            collected += 1

    return A_samples, y_samples


# ============================================================
# Main
# ============================================================

def main():
    # -----------------------------
    # Config
    # -----------------------------
    SEED = 42
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset params
    N = 30
    P_IN = 0.36
    P_OUT = 0.02
    NUM_GRAPHS = 500
    MIN_COMM_SIZE = 3

    # Training params
    NUM_CLASSES = 2
    EPOCHS = 200
    BATCH_SIZE = 16
    PCD_MINI_STEPS = 80
    RESET_PROB = 0.08
    INIT_P = 0.12
    ANNEAL_RATIO = 0.2  # 15–30% is typical; 0.2 is a safe default

    # Model params
    NODE_FEAT_DIM = NUM_CLASSES  # one-hot labels
    HIDDEN_DIM = 256
    MP_STEPS = 6
    K_HOPS_STRUCT = 3
    RESIDUAL_SCALE = 0.2

    # Optimizer
    LR = 2e-5
    WEIGHT_DECAY = 1e-4

    # Sampling params (post-training)
    N_GEN = 16           # how many graphs to plot/save
    CHAINS = 16
    BURNIN = 8000        # longer -> better mixing
    THIN = 600
    SAMPLE_INIT_P = 0.10
    SAMPLE_P_EDGE = 0.85 # after training, emphasize edges, still update labels sometimes

    # Paths

    # -----------------------------
    # Load or generate dataset
    # -----------------------------

    graphs, metadata = load_community_dataset("data/community_sbm/params_2_30_0.8_0.02_8/train.pkl")

    graphs_not_one_hot_encoded = [GraphTuple(g.adj, torch.argmax(g.node_features, axis = 1)) for g in graphs]

    dataset = CommunitySBMDataset(graphs_not_one_hot_encoded)
    print(dataset[0][0])
    print(dataset[0][1])

    show_community_graph_grid(np.random.choice(graphs, 4), cols = 2, rows = 2, layout = "spring", outdir = "gnn_figures", filename="grid_reference.png")

    # -----------------------------
    # Plot + save 2x2 real grid
    # -----------------------------
 
    print(f"[INFO] Saving 2x2 grid of REAL graphs to gnn_figures")
    
    

    # -----------------------------
    # Model + optimizer
    # -----------------------------
    model = TGNN_EBM(
        node_feat_dim=NODE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        mp_steps=MP_STEPS,
        k_hops_struct=K_HOPS_STRUCT,
        residual_scale=RESIDUAL_SCALE,
        normalize_energy=True,
        edge_barrier_weight=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.5, 0.9),
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("[INFO] Starting training...")
    persistent_chains = train_pcd_batched_joint(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        device=device,
        n_epochs=EPOCHS,
        anneal_ratio=ANNEAL_RATIO,
        batch_size=BATCH_SIZE,
        mini_steps=PCD_MINI_STEPS,
        reset_prob=RESET_PROB,
        init_p=INIT_P,
        num_classes=NUM_CLASSES,
        print_every=1,
        save_dir="gnn_figures",
    )
    print(f"[INFO] Training done. Checkpoint saved in: gnn_figures")

    # -----------------------------
    # Sample from trained model (long mixing)
    # -----------------------------
    print("[INFO] Sampling from trained EBM (joint labels+edges)...")
    A_gen, y_gen = sample_from_ebm_joint(
        model=model,
        device=device,
        N=N,
        num_classes=NUM_CLASSES,
        n_chains=CHAINS,
        burnin=BURNIN,
        thin=THIN,
        n_collect=N_GEN,
        init_edge_p=SAMPLE_INIT_P,
        p_edge=SAMPLE_P_EDGE,
    )

    # -----------------------------
    # Plot + save generated grid
    # -----------------------------

    graphs_sampled = [GraphTuple(aa, F.one_hot(yy)) for aa, yy in zip(A_gen, y_gen)]

    print(f"[INFO] Saving grid of GENERATED graphs to")

    show_community_graph_grid(np.random.choice(graphs), 4, cols = 2, rows = 2, layout = "spring", outdir="gnn_figures", filename="grid_results.png")

if __name__ == "__main__":
    main()
