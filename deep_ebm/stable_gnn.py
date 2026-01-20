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
# -----------------------------
# Model components
# -----------------------------



def sanitize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Works for:
      - [N, N]
      - [B, N, N]

    Enforces:
      - symmetry
      - zero diagonal
    Gradient-safe (no detach, no in-place ops).
    """

    # Symmetrize last two dimensions
    adj_sym = 0.5 * (adj + adj.transpose(-1, -2))

    # Zero diagonal (batched-safe)
    diag = torch.diagonal(adj_sym, dim1=-2, dim2=-1)
    adj_sym = adj_sym - torch.diag_embed(diag)

    return adj_sym


def sn_linear(in_dim, out_dim):
    return nn.utils.parametrizations.spectral_norm(nn.Linear(in_dim, out_dim))

class StableMPBlock(nn.Module):
    def __init__(self, hidden_dim, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale

        self.phi = nn.Sequential(
            sn_linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            sn_linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj):
        # adj: [N, N] or [B, N, N]
        if adj.dim() == 3:
            deg = adj.sum(-1).clamp_min(1.0)            # [B, N]
            inv_sqrt = deg.rsqrt()
            A_norm = inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)   # [B,N,N]
        else:
            deg = adj.sum(-1).clamp_min(1.0)            # [N]
            inv_sqrt = deg.rsqrt()
            A_norm = inv_sqrt[:, None] * adj * inv_sqrt[None, :]

        m = torch.matmul(A_norm, self.phi(h))
        h_next = h + self.residual_scale * m
        return self.norm(h_next)


class StableGNN_EBM(nn.Module):
    """
    Minimal SBM-capable fix:
      - adds differentiable topology features derived from adj (degree + log-degree)
      - removes overly aggressive energy normalization
      - keeps spectral norm + residual scaling for stability
      - still differentiable wrt adjacency
    """
    def __init__(self, node_attr_dim: int, hidden_dim: int = 128, mp_steps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mp_steps = mp_steps

       
        self.topo_dim = 2
        in_dim = node_attr_dim + self.topo_dim

        self.node_embed = nn.Sequential(
            sn_linear(in_dim, hidden_dim),
            nn.SiLU(),
            sn_linear(hidden_dim, hidden_dim),
        )

        
        residual_scale = min(0.25, 1.0 / max(1, mp_steps))  

        self.layers = nn.ModuleList(
            [StableMPBlock(hidden_dim, residual_scale) for _ in range(mp_steps)]
        )

        self.readout = nn.Sequential(
            sn_linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            sn_linear(hidden_dim, 1),
        )

        # --- FIX A (optional, stable): learnable temperature/scale ---
        # Starts at 1.0; the model can increase/decrease energy scale as needed.
        self.energy_scale = nn.Parameter(torch.tensor(1.0))

    def _topo_feats(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Differentiable topology features from adjacency.
        Works for both [N,N] and [B,N,N].
        Returns: [N,2] or [B,N,2]
        """
        if adj.dim() == 2:
            deg = adj.sum(-1, keepdim=True)                      # [N,1]
            deg_mean = deg.mean().clamp_min(1e-6)
            deg_norm = deg / deg_mean
            logdeg = torch.log1p(deg)
            return torch.cat([deg_norm, logdeg], dim=-1)         # [N,2]
        else:
            deg = adj.sum(-1, keepdim=True)                      # [B,N,1]
            deg_mean = deg.mean(dim=1, keepdim=True).clamp_min(1e-6)
            deg_norm = deg / deg_mean
            logdeg = torch.log1p(deg)
            return torch.cat([deg_norm, logdeg], dim=-1)         # [B,N,2]

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        node_feats = node_feats.to(dev)
        adj = adj.to(dev)

      

        topo = self._topo_feats(adj)
        x = torch.cat([node_feats, topo], dim=-1)  # [N,D+2] or [B,N,D+2]

        h = self.node_embed(x)

        for layer in self.layers:
            h = layer(h, adj)

        # Pool (batch-safe)
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        h_mean = h.mean(dim=1)
        h_var  = h.var(dim=1, unbiased=False)
        h_max  = h.max(dim=1).values

        g = torch.cat([h_mean, h_var, h_max], dim=-1)
        E = self.readout(g).squeeze(-1)

        
        E = self.energy_scale * E

        if squeeze_back:
            E = E.squeeze(0)

        return E
    


class TopologyEncoder(nn.Module):
    """
    Batched structural features from adjacency only.
    Returns:
      - unbatched: [N, F_struct]
      - batched:   [B, N, F_struct]
    """
    def __init__(
        self,
        k_hops: int = 3,
        use_logdeg: bool = True,
        lazy_alpha: float = 0.85,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.k_hops = k_hops
        self.use_logdeg = use_logdeg
        self.lazy_alpha = lazy_alpha
        self.eps = eps

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        adj: [N,N] or [B,N,N] (continuous, symmetric, no self-loops recommended)
        """
        N = adj.size(-1)

        if adj.dim() == 2:
            # [N,1]
            one = torch.ones((N, 1), device=adj.device, dtype=adj.dtype)
            deg = torch.matmul(adj, one)  # [N,1]
            deg_mean = deg.mean(dim=0, keepdim=True) + self.eps  # [1,1]

            feats = [one, deg / deg_mean]
            if self.use_logdeg:
                feats.append(torch.log1p(deg))

            deg_norm = deg + self.eps
            adj_norm = adj / deg_norm  # broadcast [N,N] / [N,1]

            x = one
            for _ in range(self.k_hops):
                x = self.lazy_alpha * torch.matmul(adj_norm, x) + (1.0 - self.lazy_alpha) * one
                feats.append(x / (x.mean(dim=0, keepdim=True) + self.eps))

            return torch.cat(feats, dim=-1)  # [N, F]

        elif adj.dim() == 3:
            B = adj.size(0)
            # [B,N,1]
            one = torch.ones((B, N, 1), device=adj.device, dtype=adj.dtype)

            deg = torch.matmul(adj, one)  # [B,N,1]
            deg_mean = deg.mean(dim=1, keepdim=True) + self.eps  # [B,1,1]

            feats = [one, deg / deg_mean]
            if self.use_logdeg:
                feats.append(torch.log1p(deg))

            deg_norm = deg + self.eps
            adj_norm = adj / deg_norm  # [B,N,N] / [B,N,1] -> [B,N,N]

            x = one
            for _ in range(self.k_hops):
                x = self.lazy_alpha * torch.matmul(adj_norm, x) + (1.0 - self.lazy_alpha) * one
                feats.append(x / (x.mean(dim=1, keepdim=True) + self.eps))  # per-graph mean

            return torch.cat(feats, dim=-1)  # [B,N,F]

        else:
            raise ValueError(f"adj must be [N,N] or [B,N,N], got shape {tuple(adj.shape)}")


class TopoMPBlock(nn.Module):
    """
    Batched message passing with residual update.
    Supports:
      - h: [N,H], adj: [N,N]
      - h: [B,N,H], adj: [B,N,N]
    """
    def __init__(self, hidden_dim: int, residual_scale: float = 0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6      # [N,1] or [B,N,1]
        adj_norm = adj / deg                             # [N,N] or [B,N,N]

        ph = self.phi(h)                                 # [N,H] or [B,N,H]
        m = torch.matmul(adj_norm, ph)                   # [N,H] or [B,N,H]

        dh = self.update(torch.cat([h, m], dim=-1))      # [N,H] or [B,N,H]
        return h + self.residual_scale * dh


class TGNN_EBM(nn.Module):
    """
    Batched EBM for topology graphs.
    Returns:
      - unbatched: scalar
      - batched:   [B]
    """
    def __init__(
        self,
        node_feat_dim: int = 1,
        hidden_dim: int = 256,
        mp_steps: int = 6,
        k_hops_struct: int = 3,
        residual_scale: float = 0.2,
        device: torch.device = torch.device("cpu"),
        normalize_energy: bool = True,
        edge_barrier_weight: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.normalize_energy = normalize_energy
        self.edge_barrier_weight = edge_barrier_weight

        self.struct_enc = TopologyEncoder(k_hops=k_hops_struct)

        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        F_struct = 2 + int(True) + k_hops_struct   # adjust if use_logdeg=False

        self.struct_embed = nn.Sequential(
            nn.Linear(F_struct, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            TopoMPBlock(hidden_dim, residual_scale)
            for _ in range(mp_steps)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        node_feats = node_feats.to(device)
        adj = sanitize_adj(adj.to(device))

        N = adj.size(-1)

        # Structural features
        s = self.struct_enc(adj)  # [N,F] or [B,N,F]

        h = self.node_embed(node_feats) + self.struct_embed(s)  # [N,H] or [B,N,H]

        for layer in self.layers:
            h = layer(h, adj)

        # Pool over nodes
        if h.dim() == 2:
            # [N,H] -> [H]
            h_mean = h.mean(dim=0)
            h_var  = h.var(dim=0, unbiased=False)
            h_max  = h.max(dim=0).values
            g = torch.cat([h_mean, h_var, h_max], dim=-1)  # [3H]
            energy = self.readout(g).squeeze()             # scalar

            if self.normalize_energy:
                energy = energy / float(N)

            if self.edge_barrier_weight > 0.0:
                barrier = (adj * (1.0 - adj)).mean()       # scalar
                energy = energy + self.edge_barrier_weight * barrier

            return energy

        elif h.dim() == 3:
            # [B,N,H] -> [B,H]
            h_mean = h.mean(dim=1)
            h_var  = h.var(dim=1, unbiased=False)
            h_max  = h.max(dim=1).values
            g = torch.cat([h_mean, h_var, h_max], dim=-1)  # [B,3H]
            energy = self.readout(g).squeeze(-1)           # [B]

            if self.normalize_energy:
                energy = energy / float(N)

            if self.edge_barrier_weight > 0.0:
                barrier = (adj * (1.0 - adj)).mean(dim=(-2, -1))  # [B]
                energy = energy + self.edge_barrier_weight * barrier

            return energy

        else:
            raise ValueError(f"node_feats/h must be 2D or 3D, got h shape {tuple(h.shape)}")




@torch.no_grad()
def edge_gibbs_ministep(adj, node_feats, model, max_delta=20.0):
    N = adj.shape[0]
    i, j = random.randrange(N), random.randrange(N)
    if i == j:
        return adj

    a, b = min(i, j), max(i, j)

    A_prime = adj.clone()
    A_prime[a, b] = 1.0 - A_prime[a, b]
    A_prime[b, a] = A_prime[a, b]

    fA = model(node_feats, adj)
    fAp = model(node_feats, A_prime)

    delta = (fAp - fA).clamp(-max_delta, max_delta)
    p = torch.sigmoid(delta).item()

    return A_prime if random.random() < p else adj



@torch.no_grad()
def node_label_gibbs_ministep(
    labels,
    adj,
    model,
    num_classes,
):
    N = labels.shape[0]
    i = random.randrange(N)

    energies = []
    for c in range(num_classes):
        new_labels = labels.clone()
        new_labels[i] = c
        one_hot = F.one_hot(new_labels, num_classes).float()
        energies.append(model(one_hot, adj))

    energies = torch.stack(energies)
    probs = torch.softmax(-energies, dim=0)
    new_c = torch.multinomial(probs, 1).item()

    labels[i] = new_c
    return labels


def blocked_gibbs_ministep(
    adj,
    labels,
    model,
    num_classes,
    p_edge=0.7,
):
    feats = F.one_hot(labels, num_classes).float()

    if random.random() < p_edge:
        adj = edge_gibbs_ministep(adj, feats, model)
    else:
        labels = node_label_gibbs_ministep(labels, adj, model, num_classes)

    return adj, labels

@torch.no_grad()
def gibbs_ministeps_edges_only(
    adj,
    feats,
    model,
    device,
    mini_steps,
    max_delta=20.0,
):
    A = sanitize_adj(adj).to(device)
    feats = feats.to(device)

    fA = model(feats, A)
    n = A.shape[0]

    for _ in range(mini_steps):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue

        a, b = min(i, j), max(i, j)

        A_prime = A.clone()
        A_prime[a, b] = 1.0 - A_prime[a, b]
        A_prime[b, a] = A_prime[a, b]

        fApr = model(feats, A_prime)
        delta = (fApr - fA).clamp(-max_delta, max_delta)
        p_acc = torch.sigmoid(delta).item()

        if random.random() < p_acc:
            A = A_prime
            fA = fApr

    return A.cpu()




def pcd_loss(
    model,
    A_data,
    labels_data,
    A_model,
    labels_model,
    num_classes,
):
    feats_data = F.one_hot(labels_data, num_classes).float()
    feats_model = F.one_hot(labels_model, num_classes).float()

    f_data = model(feats_data, A_data)
    f_model = model(feats_model, A_model)

    return -(f_data - f_model)



def atomic_torch_save(obj, path):
    """HPC-safe atomic save."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def train_one_epoch_pcd(
    model,
    dataset,
    optimizer,
    device,
    persistent_chains,
    num_classes,
    mini_steps=200,
    reset_prob=0.1,
    init_edge_p=0.1,
):
    model.train()
    total_loss = 0.0

    # init persistent chains
    if len(persistent_chains) < len(dataset):
        for A, labels in dataset:
            n = A.shape[0]
            Ar = torch.bernoulli(torch.full((n, n), init_edge_p))
            Ar = torch.triu(Ar, diagonal=1)
            Ar = Ar + Ar.T
            persistent_chains.append({
                "adj": Ar,
                "labels": labels.clone(),
            })

    for i, (A_data, labels_data) in enumerate(dataset):
        A_data = sanitize_adj(A_data).to(device)
        labels_data = labels_data.to(device)

        # reset chain occasionally
        if random.random() < reset_prob:
            n = A_data.shape[0]
            Ar = torch.bernoulli(torch.full((n, n), init_edge_p))
            Ar = torch.triu(Ar, diagonal=1)
            persistent_chains[i]["adj"] = Ar + Ar.T
            persistent_chains[i]["labels"] = labels_data.clone()

        # Gibbs sampling
        adj = persistent_chains[i]["adj"].to(device)
        labels = persistent_chains[i]["labels"].to(device)

        for _ in range(mini_steps):
            adj, labels = blocked_gibbs_ministep(
                adj, labels, model, num_classes
            )

        persistent_chains[i]["adj"] = adj.detach().cpu()
        persistent_chains[i]["labels"] = labels.detach().cpu()

        # PCD update
        optimizer.zero_grad()
        loss = pcd_loss(
            model,
            A_data,
            labels_data,
            adj,
            labels,
            num_classes,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)



def train_pcd(
    model,
    dataset,
    optimizer,
    device,
    n_epochs: int,
    num_classes: int,
    mini_steps: int = 200,
    reset_prob: float = 0.1,
    init_edge_p: float = 0.1,
    p_edge: float = 0.7,          # ← NEW
    print_every: int = 1,
    save_dir: str = None,
):
    """
    Train an EBM with Persistent Contrastive Divergence (PCD)
    using BLOCKED Gibbs (edges + node labels).
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    persistent_chains = []

    # --------------------------------------------------
    # Initialize persistent chains
    # --------------------------------------------------
    if len(persistent_chains) < len(dataset):
        for A_data, labels_data in dataset:
            n = A_data.shape[0]

            A_init = torch.bernoulli(
                torch.full((n, n), init_edge_p)
            )
            A_init = torch.triu(A_init, diagonal=1)
            A_init = A_init + A_init.T

            persistent_chains.append({
                "adj": A_init,
                "labels": labels_data.clone(),
            })

    # --------------------------------------------------
    # Training epochs
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        start = time.time()

        model.train()
        total_loss = 0.0
        total_f_data = 0.0
        total_f_model = 0.0

        for i, (A_data, labels_data) in enumerate(dataset):
            A_data = sanitize_adj(A_data).to(device)
            labels_data = labels_data.to(device)

            # ------------------------------------------
            # Occasionally reset chain
            # ------------------------------------------
            if random.random() < reset_prob:
                n = A_data.shape[0]
                A_reset = torch.bernoulli(
                    torch.full((n, n), init_edge_p)
                )
                A_reset = torch.triu(A_reset, diagonal=1)
                persistent_chains[i]["adj"] = A_reset + A_reset.T
                persistent_chains[i]["labels"] = labels_data.clone()

            # ------------------------------------------
            # Load persistent state
            # ------------------------------------------
            adj = persistent_chains[i]["adj"].to(device)
            labels = persistent_chains[i]["labels"].to(device)

            # ------------------------------------------
            # BLOCKED Gibbs sampling
            # ------------------------------------------
            for _ in range(mini_steps):
                adj, labels = blocked_gibbs_ministep(
                    adj=adj,
                    labels=labels,
                    model=model,
                    num_classes=num_classes,
                    p_edge=p_edge,
                )

            persistent_chains[i]["adj"] = adj.detach().cpu()
            persistent_chains[i]["labels"] = labels.detach().cpu()

            # ------------------------------------------
            # PCD update
            # ------------------------------------------
            feats_data = F.one_hot(labels_data, num_classes).float()
            feats_model = F.one_hot(labels, num_classes).float()

            f_data = model(feats_data, A_data)
            f_model = model(feats_model, adj)

            loss = -(f_data - f_model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_data += f_data.item()
            total_f_model += f_model.item()

        avg_loss = total_loss / len(dataset)
        elapsed = time.time() - start

        if epoch % print_every == 0:
            print(
                f"[Epoch {epoch:03d}/{n_epochs}] "
                f"loss={avg_loss:.4f} | "
                f"f_data={total_f_data / len(dataset):.3f} | "
                f"f_model={total_f_model / len(dataset):.3f} | "
                f"time={elapsed:.1f}s"
            )

    # --------------------------------------------------
    # Final save
    # --------------------------------------------------
    if save_dir is not None:
        final_ckpt = {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "persistent_chains": persistent_chains,
            "loss": avg_loss,
            "p_edge": p_edge,
        }
        path = os.path.join(save_dir, "final_model.pt")
        atomic_torch_save(final_ckpt, path)
        print(f"[INFO] Final model saved to {path}")

    return persistent_chains




def train_pcd_edge_only(
    model,
    dataset,
    optimizer,
    device,
    n_epochs: int,
    mini_steps: int = 200,
    reset_prob: float = 0.1,
    init_edge_p: float = 0.1,
    print_every: int = 1,
    save_dir: str = None,
):
    """
    PCD training for EDGE-ONLY EBMs.
    Node features are fixed and given by the dataset.
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    persistent_chains = []

    # --------------------------------------------------
    # Initialize persistent chains (adjacency only)
    # --------------------------------------------------
    if len(persistent_chains) < len(dataset):
        for A_data, _ in dataset:
            n = A_data.shape[0]
            Ar = torch.bernoulli(
                torch.full((n, n), init_edge_p)
            )
            Ar = torch.triu(Ar, diagonal=1)
            persistent_chains.append(Ar + Ar.T)

    # --------------------------------------------------
    # Training epochs
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        start = time.time()

        model.train()
        total_loss = 0.0
        total_f_data = 0.0
        total_f_model = 0.0

        for i, (A_data, feats) in enumerate(dataset):
            A_data = sanitize_adj(A_data).to(device)
            feats = feats.to(device)

            # ------------------------------------------
            # Occasionally reset chain
            # ------------------------------------------
            if random.random() < reset_prob:
                n = A_data.shape[0]
                Ar = torch.bernoulli(
                    torch.full((n, n), init_edge_p)
                )
                Ar = torch.triu(Ar, diagonal=1)
                persistent_chains[i] = Ar + Ar.T

            # ------------------------------------------
            # Gibbs sampling (edges only)
            # ------------------------------------------
            adj = persistent_chains[i].to(device)
            adj = gibbs_ministeps_edges_only(
                adj, feats, model, device, mini_steps
            )
            persistent_chains[i] = adj.detach().cpu()

            # ------------------------------------------
            # PCD update
            # ------------------------------------------
            f_data = model(feats, A_data)
            f_model = model(feats, adj)

            loss = -(f_data - f_model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_data += f_data.item()
            total_f_model += f_model.item()

        avg_loss = total_loss / len(dataset)
        elapsed = time.time() - start

        if epoch % print_every == 0:
            print(
                f"[Epoch {epoch:03d}/{n_epochs}] "
                f"loss={avg_loss:.4f} | "
                f"f_data={total_f_data / len(dataset):.3f} | "
                f"f_model={total_f_model / len(dataset):.3f} | "
                f"time={elapsed:.1f}s"
            )

    # --------------------------------------------------
    # Final save
    # --------------------------------------------------
    if save_dir is not None:
        final_ckpt = {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "persistent_chains": persistent_chains,
            "loss": avg_loss,
        }
        path = os.path.join(save_dir, "final_model.pt")
        atomic_torch_save(final_ckpt, path)
        print(f"[INFO] Final model saved to {path}")

    return persistent_chains
