import math
import random
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import os



def generate_2community_labeled_sbm_fixed_n(
    num_graphs: int,
    n: int,
    p_in: float,
    p_out: float,
    seed: int = 0,
    min_comm_size: int = 3,
):
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_graphs):
        # --- random split ---
        low = min_comm_size
        high = n - min_comm_size
        n1 = int(rng.integers(low, high + 1))
        n2 = n - n1

        # OLD CODE - BROKEN: Labels didn't match communities after permutation
        # labels = np.zeros(n, dtype=np.int64)
        # labels[n1:] = 1

        probs = [[p_in, p_out],
                 [p_out, p_in]]

        s = int(rng.integers(0, 10**9))
        G = nx.stochastic_block_model([n1, n2], probs, seed=s)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # --- random permutation ---
        perm = rng.permutation(n)
        G = nx.relabel_nodes(G, {i: int(perm[i]) for i in range(n)})
        
        # NEW CODE - FIXED: Create proper community assignment that matches SBM structure
        # The SBM creates communities [n1, n2], so we need to track which nodes belong to which community
        community_assignment = np.zeros(n, dtype=np.int64)
        community_assignment[:n1] = 0  # First n1 nodes belong to community 0
        community_assignment[n1:] = 1  # Remaining n2 nodes belong to community 1
        
        # Apply the same permutation to keep labels aligned with communities
        labels = community_assignment[perm]
        
        # OLD CODE - BROKEN: This shuffled labels randomly, breaking the community-label relationship
        # labels = labels[perm]

        samples.append({
            "G": G,
            "labels": labels
        })

    return samples



@torch.no_grad()
def gibbs_ministeps_edges_batch(
    A_batch: torch.Tensor,     # [B, N, N]
    feats_batch: torch.Tensor, # [B, N, D]
    model,
    device,
    mini_steps: int,
):
    B, N, _ = A_batch.shape
    A = A_batch.to(device)
    feats = feats_batch.to(device)

    fA = model(feats, A)   # [B]

    for _ in range(mini_steps):
        i = torch.randint(0, N, (B,), device=device)
        j = torch.randint(0, N, (B,), device=device)

        mask = i != j
        if not mask.any():
            continue

        a = torch.minimum(i, j)
        b = torch.maximum(i, j)

        A_prop = A.clone()
        idx = torch.arange(B, device=device)[mask]
        A_prop[idx, a[mask], b[mask]] = 1.0 - A_prop[idx, a[mask], b[mask]]
        A_prop[idx, b[mask], a[mask]] = A_prop[idx, a[mask], b[mask]]

        f_prop = model(feats, A_prop)

        p_acc = torch.sigmoid(f_prop - fA)
        accept = torch.rand(B, device=device) < p_acc

        A[accept] = A_prop[accept]
        fA[accept] = f_prop[accept]

    return A


@torch.no_grad()
def gibbs_step_labels_batch(
    labels_batch: torch.Tensor,  # [B, N]
    A_batch: torch.Tensor,       # [B, N, N]
    model,
    num_classes: int,
):
    B, N = labels_batch.shape
    labels = labels_batch.clone()

    for b in range(B):
        i = random.randrange(N)

        energies = []
        for c in range(num_classes):
            tmp = labels[b].clone()
            tmp[i] = c
            feats = F.one_hot(tmp, num_classes).float()
            energies.append(model(feats, A_batch[b]))

        energies = torch.stack(energies)
        probs = torch.softmax(energies, dim=0)   # POSITIVE ENERGY
        new_c = torch.multinomial(probs, 1).item()

        labels[b, i] = new_c

    return labels


@torch.no_grad()
def blocked_gibbs_ministeps_batch(
    A_batch,
    labels_batch,
    model,
    device,
    mini_steps,
    num_classes,
    p_edge=0.6,
):
    A = A_batch.to(device)
    labels = labels_batch.to(device)

    for _ in range(mini_steps):
        if random.random() < p_edge:
            feats = F.one_hot(labels, num_classes).float()
            A = gibbs_ministeps_edges_batch(
                A, feats, model, device, mini_steps=1
            )
        else:
            labels = gibbs_step_labels_batch(
                labels, A, model, num_classes
            )

    return A.detach(), labels.detach()


def edge_prob(epoch, anneal_epochs, p_edge_start=0.4, p_edge_end=0.8):
    if epoch <= anneal_epochs:
        return p_edge_start + (p_edge_end - p_edge_start) * epoch / anneal_epochs
    else:
        return p_edge_end



def train_pcd_batched_joint(
    model,
    dataset,
    optimizer,
    device,
    n_epochs: int,
    anneal_ratio: float = 0.3,
    batch_size: int = 16,
    mini_steps: int = 50,
    reset_prob: float = 0.05,
    init_p: float = 0.1,
    num_classes: int = 2,
    p_edge: float = 0.6,
    print_every: int = 1,
    save_dir: str = None,
):
    """
    Batched PCD for joint (A, labels) EBM with POSITIVE energy convention.
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model.train()

    N = dataset[0][0].shape[0]
    num_batches = math.ceil(len(dataset) / batch_size)

    # --------------------------------------------------
    # Initialize persistent chains
    # --------------------------------------------------
    persistent_chains = []
    for A_data, labels_data in dataset:
        Ar = torch.bernoulli(torch.full((N, N), init_p))
        Ar = torch.triu(Ar, diagonal=1)
        Ar = Ar + Ar.T

        persistent_chains.append({
            "adj": Ar,
            "labels": labels_data.clone(),
        })

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    anneal_epochs = int(n_epochs * anneal_ratio)
    for epoch in range(1, n_epochs + 1):
        start = time.time()

        perm = torch.randperm(len(dataset))
        total_loss = 0.0
        total_f_data = 0.0
        total_f_model = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            B = len(idx)

            # -----------------------------
            # Load batch
            # -----------------------------
            A_data = torch.stack([dataset[i][0] for i in idx]).to(device)
            labels_data = torch.stack([dataset[i][1] for i in idx]).to(device)

            A_model = torch.stack(
                [persistent_chains[i]["adj"] for i in idx]
            ).to(device)

            labels_model = torch.stack(
                [persistent_chains[i]["labels"] for i in idx]
            ).to(device)

            # -----------------------------
            # Optional reset
            # -----------------------------
            reset_mask = torch.rand(B, device=device) < reset_prob
            if reset_mask.any():
                Ar = torch.bernoulli(
                    torch.full((B, N, N), init_p, device=device)
                )
                Ar = torch.triu(Ar, diagonal=1)
                A_model[reset_mask] = Ar[reset_mask] + Ar[reset_mask].transpose(1, 2)
                labels_model[reset_mask] = labels_data[reset_mask]

            # -----------------------------
            # BLOCKED Gibbs
            # -----------------------------

            p_edge = edge_prob(epoch, anneal_epochs, p_edge_start=0.35, p_edge_end=0.8)
            A_model, labels_model = blocked_gibbs_ministeps_batch(
                A_model,
                labels_model,
                model,
                device,
                mini_steps,
                num_classes,
                p_edge=p_edge,
            )

            # store back
            for k, i in enumerate(idx):
                persistent_chains[i]["adj"] = A_model[k].cpu()
                persistent_chains[i]["labels"] = labels_model[k].cpu()

            # -----------------------------
            # PCD gradient
            # -----------------------------
            feats_data  = F.one_hot(labels_data, num_classes).float()
            feats_model = F.one_hot(labels_model, num_classes).float()

            f_data  = model(feats_data, A_data)
            f_model = model(feats_model, A_model)

            loss = -(f_data.mean() - f_model.mean())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_f_data += f_data.mean().item()
            total_f_model += f_model.mean().item()

        avg_loss = total_loss / num_batches
        elapsed = time.time() - start

        if epoch % print_every == 0:
            print(
                f"[Epoch {epoch:03d}/{n_epochs}] "
                f"loss={avg_loss:.4f} | "
                f"f_data={total_f_data / num_batches:.3f} | "
                f"f_model={total_f_model / num_batches:.3f} | "
                f"p_edge={p_edge:.3f} | "
                f"time={elapsed:.1f}s"
            )

    # --------------------------------------------------
    # Final save
    # --------------------------------------------------
    if save_dir is not None:
        ckpt = {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "persistent_chains": persistent_chains,
            "loss": avg_loss,
            "p_edge": p_edge,
        }
        path = os.path.join(save_dir, "final_model.pt")
        atomic_torch_save(ckpt, path)
        print(f"[INFO] Final model saved to {path}")

    return persistent_chains




def graph_from_adj_labels(A, labels=None, threshold=0.5):
    """
    A: [N,N] tensor or numpy
    labels: [N] tensor or numpy (optional)
    """
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
    if labels is not None and hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    A_bin = (A > threshold).astype(np.int32)
    G = nx.from_numpy_array(A_bin)

    if labels is not None:
        for i, y in enumerate(labels):
            G.nodes[i]["label"] = int(y)

    return G



import matplotlib.pyplot as plt

def plot_graph(
    G,
    labels=None,
    layout="spring",
    seed=0,
    node_size=300,
    edge_alpha=0.6,
    ax=None,
    title=None,
):
    """
    layout: "spring" | "kamada_kawai" | "spectral"
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    if labels is None:
        labels = nx.get_node_attributes(G, "label")
        if labels:
            labels = np.array([labels[i] for i in range(len(labels))])
        else:
            labels = np.zeros(G.number_of_nodes(), dtype=int)

    # layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError(f"Unknown layout {layout}")

    colors = ["tab:blue" if y == 0 else "tab:orange" for y in labels]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=node_size,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=edge_alpha,
        ax=ax,
    )

    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)



def plot_graph_grid(
    A_list,
    y_list,
    rows=2,
    cols=4,
    layout="spring",
    seed=0,
    figsize=(12, 6),
):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for k, ax in enumerate(axes.flat):
        if k >= len(A_list):
            ax.axis("off")
            continue

        G = graph_from_adj_labels(A_list[k], y_list[k])
        plot_graph(G, y_list[k], layout=layout, seed=seed + k, ax=ax)

    plt.tight_layout()
    plt.show()
