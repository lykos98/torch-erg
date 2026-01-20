"""
GNN-EBM implementation (PyTorch)

Implements architecture and training / sampling procedures described in
"Graph Generation with Energy-Based Models" (Liu et al., 2020) PDF the user provided.

Features implemented:
- GNN-EBM model: dot-product attention message passing, 6 MP steps,
  MLP readout (3 layers, 1024 hidden) producing a scalar energy f(G;theta).
- Losses: Pseudolikelihood, Conditional NCE (with symmetric corruption),
  Persistent Contrastive Divergence (PCD) using Gibbs sampling.
- Corruption procedure for CNCE (Algorithm 1) using Beta noise.
- Gibbs sampler for graphs (Algorithm 2) and PCD persistent chains.
- Generation: greedy energy minimization + optional Gibbs refinement.
- Evaluation: degree and clustering MMD between generated graphs and dataset.

Notes / assumptions:
- This implementation operates on single graphs (variable |V|). For efficiency
  you may want to batch graphs with padding or use PyG / DGL for large-scale runs.
- "Orbit" statistics (graphlet-orbit counts) used in the paper are not
  implemented here; a placeholder is provided if you want to add a graphlet
  library (e.g., ORCA) to compute orbit counts.
- Dataset loader expects a list of networkx.Graph objects. See `GraphDataset`.

Run as a module or import functions in your own training harness.
"""

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologyEncoder(nn.Module):
    """
    Builds structural node features from adjacency only.
    Outputs per-node structural features, permutation-equivariant, differentiable in A.
    """
    def __init__(self, k_hops: int = 3, use_logdeg: bool = True):
        super().__init__()
        self.k_hops = k_hops
        self.use_logdeg = use_logdeg

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        # adj: [N,N] (can be continuous)
        N = adj.shape[0]
        one = torch.ones((N, 1), device=adj.device, dtype=adj.dtype)

        # degree
        deg = adj @ one                       # [N,1]
        deg_mean = deg.mean() + 1e-6          # stable, differentiable

        feats = [one, deg / deg_mean]
        if self.use_logdeg:
            feats.append(torch.log1p(deg))

        # k-hop diffusion features: A^t 1, t=2..k+1 (or using normalized A if you prefer)
        x = deg  # = A 1
        for _ in range(self.k_hops):
            x = adj @ x
            feats.append(x / (x.mean() + 1e-6))

        return torch.cat(feats, dim=-1)       # [N, F_struct]


class TopoMPBlock(nn.Module):
    """
    Topology-driven message passing:
      m = A_norm @ phi(h)
      h <- GRU(m, h)   (gated update)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Normalize adjacency for stability across datasets/densities
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj / deg

        m = adj_norm @ self.phi(h)
        h_next = self.gru(m, h)
        return h_next


class TGNN_EBM(nn.Module):
    """
    Energy model f(G;θ) that heavily prioritizes topology while still accepting node features.

    forward(node_feats, adj) -> scalar
    - node_feats can be constant ones
    - topology features are derived from adj and injected explicitly
    - permutation invariant graph-level energy
    - differentiable w.r.t. adj
    """
    def __init__(
        self,
        node_feat_dim: int = 16,
        hidden_dim: int = 256,
        mp_steps: int = 6,
        k_hops_struct: int = 3,
        device: torch.device = torch.device("cpu"),
        temperature : float = 30.0,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.struct_enc = TopologyEncoder(k_hops=k_hops_struct)

        # We'll infer struct feature dimension at runtime; easiest is to use LazyLinear
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

        self.struct_embed = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

        self.layers = nn.ModuleList([TopoMPBlock(hidden_dim).to(device) for _ in range(mp_steps)])

        # Invariant readout (mean + var + max)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        node_feats = node_feats.to(self.device)
        adj = adj.to(self.device)

        # explicit topology features
        s = self.struct_enc(adj)          # [N, F_struct]
        h = self.node_embed(node_feats) + self.struct_embed(s)

        for layer in self.layers:
            h = layer(h, adj)

        # invariant pooling
        h_mean = h.mean(dim=0)
        h_var = h.var(dim=0, unbiased=False)
        h_max = h.max(dim=0).values
        g = torch.cat([h_mean, h_var, h_max], dim=-1)

        return self.readout(g).squeeze()



class ConstFeatWrapper(nn.Module):
    def __init__(self, base, node_feat_dim):
        super().__init__()
        self.base = base
        self.node_feat_dim = node_feat_dim

    def forward(self, adj):
        n = adj.shape[0]
        feats = torch.ones((n, self.node_feat_dim),
                           device=adj.device,
                           dtype=adj.dtype)
        return self.base(feats, adj)


# -----------------------------
# Dataset wrapper
# -----------------------------

class GraphDataset(Dataset):
    def __init__(self, graphs: List[nx.Graph], max_nodes: Optional[int] = None):
        self.graphs = graphs
        self.max_nodes = max_nodes or max(g.number_of_nodes() for g in graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        G = self.graphs[idx]
        n = G.number_of_nodes()
        A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        # padded one-hot features: (n × max_nodes)
        feats = np.ones((n, 1), dtype=np.float32)
        return torch.from_numpy(A).float(), torch.from_numpy(feats).float()


# -----------------------------
# Utilities: flips, corruption, Gibbs sampling
# -----------------------------

def flip_edge(adj: torch.Tensor, i: int, j: int) -> torch.Tensor:
    A = adj.clone()
    A[i, j] = 1.0 - A[i, j]
    A[j, i] = A[i, j]
    return A


def cnce_corrupt(adj: torch.Tensor, alpha=1.0, beta=20.0) -> torch.Tensor:
    """Algorithm 1: independently flip each upper-triangular edge with probability p ~ Beta(alpha,beta)
    Returns corrupted adjacency matrix tensor (symmetric)
    """
    n = adj.shape[0]
    p = np.random.beta(alpha, beta)
    A = adj.clone()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                A[i, j] = 1.0 - A[i, j]
                A[j, i] = A[i, j]
    return A


def gibbs_step(adj: torch.Tensor, model: nn.Module, node_feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    """One full Gibbs sweep over all upper-triangular edges, performing the "mini-step" per paper.
    Follows Algorithm 2 but for a single full step.
    """
    n = adj.shape[0]
    A = adj.clone()
    for j in range(n):
        for k in range(j + 1, n):
            A_prime = A.clone()
            A_prime[j, k] = 1.0 - A_prime[j, k]
            A_prime[k, j] = A_prime[j, k]
            # compute p = exp(f(A)) / (exp(f(A)) + exp(f(A')))
            with torch.no_grad():
                fA = model(node_feats.to(device), A.to(device))
                fApr = model(node_feats.to(device), A_prime.to(device))
                # numerically stable sigmoid
                p = torch.sigmoid(fA - fApr).item()
            if random.random() > p:
                A = A_prime
    return A


def gibbs_ministeps(adj: torch.Tensor, model: nn.Module, node_feats: torch.Tensor, device: torch.device, mini_steps: int) -> torch.Tensor:
    # run `mini_steps` individual mini-steps: sample a random pair each time
    # its formulation is equal to the standard MH one
    n = adj.shape[0]
    A = adj.clone()
    for _ in range(mini_steps):
        i = random.randrange(0, n)
        j = random.randrange(0, n)
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        A_prime = A.clone()
        A_prime[a, b] = 1.0 - A_prime[a, b]
        A_prime[b, a] = A_prime[a, b]
        with torch.no_grad():
            fA = model(node_feats.to(device), A.to(device))

            fApr = model(node_feats.to(device), A_prime.to(device))

            p = torch.sigmoid(fA - fApr).item()
        if random.random() > p:
            A = A_prime
    return A



def gibbs_ministeps_batch(
    adj: torch.Tensor,
    model: nn.Module,
    node_feats: torch.Tensor,
    device: torch.device,
    mini_steps: int,
    n_samples: int = 1,
):
    """
    Run Gibbs mini-steps in parallel for multiple chains.
    
    Args:
        adj: torch.Tensor (n x n) initial adjacency
        model: GNN_EBM
        node_feats: torch.Tensor (n x d) node features
        device: torch.device
        mini_steps: number of Gibbs updates per chain
        n_samples: number of parallel chains
    
    Returns:
        torch.Tensor (n_samples x n x n) final adjacency matrices
    """
    n = adj.shape[0]

    # replicate initial adjacency for all chains
    A_batch = adj.unsqueeze(0).repeat(n_samples, 1, 1).to(device)
    feats = node_feats.to(device)

    # precompute initial energies
    with torch.no_grad():
        fA = torch.stack([model(feats, A_batch[i]) for i in range(n_samples)])

    for _ in range(mini_steps):
        # pick random edge indices for each chain
        i = torch.randint(0, n, (n_samples,), device=device)
        j = torch.randint(0, n, (n_samples,), device=device)
        mask = (i != j)  # skip self-loops
        if not mask.any():
            continue

        # create proposals by flipping (vectorized)
        A_prime = A_batch.clone()
        a = torch.minimum(i, j)
        b = torch.maximum(i, j)
        for k in torch.nonzero(mask).flatten():
            A_prime[k, a[k], b[k]] = 1.0 - A_prime[k, a[k], b[k]]
            A_prime[k, b[k], a[k]] = A_prime[k, a[k], b[k]]

        with torch.no_grad():
            fApr = torch.stack([model(feats, A_prime[k]) for k in range(n_samples)])

        # acceptance probabilities
        p = torch.sigmoid(fA - fApr)

        # decide which chains accept
        rand = torch.rand(n_samples, device=device)
        accept = rand > p

        # update accepted chains
        for k in torch.nonzero(accept).flatten():
            A_batch[k] = A_prime[k]
            fA[k] = fApr[k]

    return A_batch.cpu()


# -----------------------------
# Losses
# -----------------------------

def pseudolikelihood_loss(model: nn.Module, adj: torch.Tensor, node_feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    # sum over upper-triangular entries of log p(A_ij | A_{\i j})
    n = adj.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            A_prime = adj.clone()
            A_prime[i, j] = 1.0 - A_prime[i, j]
            A_prime[j, i] = A_prime[i, j]
            fA = model(node_feats.to(device), adj.to(device))
            fApr = model(node_feats.to(device), A_prime.to(device))
            # p(A_ij) = exp(fA) / (exp(fA) + exp(fApr))
            logp = F.logsigmoid(fA - fApr)
            # We want negative log-likelihood -> minimize -sum log p
            loss = loss - logp
    return loss


def cnce_loss(model: nn.Module, adj: torch.Tensor, node_feats: torch.Tensor, corrupt_fn, device: torch.device) -> torch.Tensor:
    # symmetric corruption so log terms cancel; F = f(x) - f(x')
    x = adj
    xprime = corrupt_fn(adj)
    f_x = model(node_feats.to(device), x.to(device))
    f_xp = model(node_feats.to(device), xprime.to(device))
    F = f_x - f_xp
    loss = torch.log1p(torch.exp(-F))  # log(1 + exp(-F))
    return loss


def pcd_loss(model: nn.Module, data_adj: torch.Tensor, node_feats: torch.Tensor, model_sample_adj: torch.Tensor, device: torch.device) -> torch.Tensor:
    # gradient approx: grad f(x_data) - grad f(x_model) -> equivalent to maximize f(data) - f(model_sample)
    f_data = model(node_feats.to(device), data_adj.to(device))
    f_model = model(node_feats.to(device), model_sample_adj.to(device))
    # want to maximize f_data - E[f_model]; minimize -(f_data - f_model)
    loss = -(f_data - f_model)
    return loss

# -----------------------------
# Evaluation: MMD for degree and clustering
# -----------------------------

def graph_degree_hist(G: nx.Graph) -> np.ndarray:
    degs = np.array([d for _, d in G.degree()])
    # return sorted degree vector
    return np.sort(degs)


def graph_clustering_vec(G: nx.Graph) -> np.ndarray:
    # per-node clustering coefficients
    clus = np.array(list(nx.clustering(G).values()))
    return np.sort(clus)


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Unbiased MMD^2 estimator with RBF kernel.
    X: [m, d]
    Y: [n, d]
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    m, n = X.shape[0], Y.shape[0]

    def rbf_kernel(A, B):
        sq_norms_A = np.sum(A**2, axis=1, keepdims=True)
        sq_norms_B = np.sum(B**2, axis=1, keepdims=True)
        dists = sq_norms_A - 2 * A @ B.T + sq_norms_B.T
        return np.exp(-dists / (2 * sigma**2))

    Kxx = rbf_kernel(X, X)
    Kyy = rbf_kernel(Y, Y)
    Kxy = rbf_kernel(X, Y)

    # remove diagonal for unbiased estimate
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd = (
        Kxx.sum() / (m * (m - 1))
        + Kyy.sum() / (n * (n - 1))
        - 2 * Kxy.mean()
    )
    return float(mmd)



def degree_variance(G: nx.Graph):
    deg = np.array([d for _, d in G.degree()])
    return float(deg.var())

def degree_gini(G: nx.Graph):
    deg = np.array([d for _, d in G.degree()], dtype=np.float64)
    if deg.sum() == 0:
        return 0.0
    deg = np.sort(deg)
    n = len(deg)
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1).dot(deg) / (n * deg.sum()))

def triangle_count(G: nx.Graph):
    # each triangle counted 3 times by nx.triangles
    return float(sum(nx.triangles(G).values()) / 3)

def avg_path_length_safe(G: nx.Graph):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    # largest connected component
    cc = max(nx.connected_components(G), key=len)
    return nx.average_shortest_path_length(G.subgraph(cc))

def scalar_mmd(x, y, sigma=1.0):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    return compute_mmd(x, y, sigma=sigma)

# -------------------------------------------------
# Extended evaluation
# -------------------------------------------------

def evaluate_mmd_generated(
    dataset_graphs: List[nx.Graph],
    generated_graphs: List[nx.Graph],
    sigma: float = 1.0,
):
    """
    Distributional evaluation via MMD over multiple graph statistics.
    """

    # -------------------------
    # Degree distribution MMD
    # -------------------------
    deg_real = [graph_degree_hist(G) for G in dataset_graphs]
    deg_gen  = [graph_degree_hist(G) for G in generated_graphs]

    maxlen = max(
        max(len(v) for v in deg_real),
        max(len(v) for v in deg_gen),
    )

    def pad(v):
        return np.pad(v, (0, maxlen - len(v)))

    deg_real = np.array([pad(v) for v in deg_real])
    deg_gen  = np.array([pad(v) for v in deg_gen])

    mmd_deg = compute_mmd(deg_real, deg_gen, sigma=sigma)

    # -------------------------
    # Clustering distribution MMD
    # -------------------------
    clus_real = [graph_clustering_vec(G) for G in dataset_graphs]
    clus_gen  = [graph_clustering_vec(G) for G in generated_graphs]

    maxlen = max(
        max(len(v) for v in clus_real),
        max(len(v) for v in clus_gen),
    )

    clus_real = np.array([pad(v) for v in clus_real])
    clus_gen  = np.array([pad(v) for v in clus_gen])

    mmd_clus = compute_mmd(clus_real, clus_gen, sigma=sigma)

    # -------------------------
    # Scalar structural statistics (MMD over scalars)
    # -------------------------
    degvar_real = [degree_variance(G) for G in dataset_graphs]
    degvar_gen  = [degree_variance(G) for G in generated_graphs]

    gini_real = [degree_gini(G) for G in dataset_graphs]
    gini_gen  = [degree_gini(G) for G in generated_graphs]

    tri_real = [triangle_count(G) for G in dataset_graphs]
    tri_gen  = [triangle_count(G) for G in generated_graphs]

    apl_real = [avg_path_length_safe(G) for G in dataset_graphs]
    apl_gen  = [avg_path_length_safe(G) for G in generated_graphs]

    mmd_degvar = scalar_mmd(degvar_real, degvar_gen, sigma)
    mmd_gini   = scalar_mmd(gini_real, gini_gen, sigma)
    mmd_tri    = scalar_mmd(tri_real, tri_gen, sigma)
    mmd_apl    = scalar_mmd(apl_real, apl_gen, sigma)

    # -------------------------
    # Return all metrics
    # -------------------------
    return {
        "degree_mmd": mmd_deg,
        "clustering_mmd": mmd_clus,
        "degvar_mmd": mmd_degvar,
        "gini_mmd": mmd_gini,
        "triangles_mmd": mmd_tri,
        "apl_mmd": mmd_apl,
    }



# ============================================================
# Graph utilities
# ============================================================

def sanitize_adj(A):
    A = A.clone()
    A.fill_diagonal_(0.0)
    return 0.5 * (A + A.T)


# ============================================================
# Gibbs sampling
# ============================================================

@torch.no_grad()
def gibbs_ministeps(adj, feats, model, device, mini_steps):
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
        p_acc = torch.sigmoid(fApr - fA).item()

        if random.random() < p_acc:
            A = A_prime
            fA = fApr

    return A.cpu()


# ============================================================
# Losses
# ============================================================

def pcd_loss(model, A_data, feats, A_model, device, reg_weight=1e-4):
    A_data = sanitize_adj(A_data).to(device)
    A_model = sanitize_adj(A_model).to(device)
    feats = feats.to(device)

    f_data = model(feats, A_data)
    f_model = model(feats, A_model)

    loss = -(f_data - f_model)
    loss = loss + reg_weight * (f_data**2 + f_model**2)
    return loss


# ============================================================
# Training loop
# ============================================================

def train_one_epoch_pcd(
    model,
    dataset,
    optimizer,
    device,
    persistent_chains,
    mini_steps=200,
    reset_prob=0.1,
    init_p=0.1,
):
    model.train()
    total_loss = 0.0

    if len(persistent_chains) < len(dataset):
        for A, _ in dataset:
            n = A.shape[0]
            Ar = torch.bernoulli(torch.full((n, n), init_p))
            Ar = torch.triu(Ar, diagonal=1)
            Ar = Ar + Ar.T
            persistent_chains.append(Ar)

    for i, (A, feats) in enumerate(dataset):
        A = A.to(device)
        feats = feats.to(device)

        if random.random() < reset_prob:
            n = A.shape[0]
            Ar = torch.bernoulli(torch.full((n, n), init_p))
            Ar = torch.triu(Ar, diagonal=1)
            persistent_chains[i] = Ar + Ar.T

        chainA = persistent_chains[i].to(device)
        chainA = gibbs_ministeps(chainA, feats, model, device, mini_steps)
        persistent_chains[i] = chainA

        optimizer.zero_grad()
        loss = pcd_loss(model, A, feats, chainA, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)





def atomic_torch_save(obj, path):
    """HPC-safe atomic save."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def train_pcd(
    model,
    dataset,
    optimizer,
    device,
    n_epochs: int,
    mini_steps: int = 200,
    reset_prob: float = 0.1,
    init_p: float = 0.1,
    print_every: int = 1,
    save_dir: str = None,
):
    """
    Train an EBM with Persistent Contrastive Divergence (PCD).

    Saves (ONLY at the end):
      - model state_dict
      - optimizer state_dict
      - persistent chains
      - final epoch index
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    persistent_chains = []

    for epoch in range(1, n_epochs + 1):
        start = time.time()

        model.train()
        total_loss = 0.0
        total_f_data = 0.0
        total_f_model = 0.0

        # Initialize persistent chains if needed
        if len(persistent_chains) < len(dataset):
            for A, _ in dataset:
                n = A.shape[0]
                Ar = torch.bernoulli(torch.full((n, n), init_p))
                Ar = torch.triu(Ar, diagonal=1)
                Ar = Ar + Ar.T
                persistent_chains.append(Ar)

        for i, (A, feats) in enumerate(dataset):
            A = A.to(device)
            feats = feats.to(device)

            # Occasionally reset chain
            if torch.rand(1).item() < reset_prob:
                n = A.shape[0]
                Ar = torch.bernoulli(torch.full((n, n), init_p))
                Ar = torch.triu(Ar, diagonal=1)
                persistent_chains[i] = Ar + Ar.T

            # Gibbs update
            chainA = persistent_chains[i].to(device)
            chainA = gibbs_ministeps(chainA, feats, model, device, mini_steps)
            persistent_chains[i] = chainA

            # Energies
            f_data = model(feats, A)
            f_model = model(feats, chainA.to(device))

            optimizer.zero_grad()
            loss = -(f_data - f_model)
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

    # -------------------------
    # Final save (ONLY ONCE)
    # -------------------------
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





@torch.no_grad()
def gibbs_ministeps_batch(
    A_batch: torch.Tensor,     # [B, N, N]
    feats_batch: torch.Tensor, # [B, N, D]
    model,
    device,
    mini_steps: int,
):
    """
    Batched single-edge Metropolis-Hastings Gibbs sampler.
    Each batch element is an independent chain.
    """
    B, N, _ = A_batch.shape
    A = A_batch.to(device)
    feats = feats_batch.to(device)

    # initial energies
    fA = model(feats, A)   # [B]

    for _ in range(mini_steps):
        # sample random edge per chain
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

    return A.detach()


def train_pcd_batched(
    model,
    dataset,
    optimizer,
    device,
    n_epochs: int,
    batch_size: int = 16,
    mini_steps: int = 50,
    reset_prob: float = 0.05,
    init_p: float = 0.1,
    print_every: int = 1,
    save_dir: str = None,
):
    """
    Batched Persistent Contrastive Divergence (PCD).
    Chains are independent but updated in parallel.
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model.train()

    N = dataset[0][0].shape[0]
    D = dataset[0][1].shape[1]

    # --------------------------------------------------
    # Initialize persistent chains
    # --------------------------------------------------
    persistent_chains = []
    for A, _ in dataset:
        Ar = torch.bernoulli(torch.full((N, N), init_p))
        Ar = torch.triu(Ar, diagonal=1)
        persistent_chains.append(Ar + Ar.T)

    num_batches = math.ceil(len(dataset) / batch_size)

    for epoch in range(1, n_epochs + 1):
        start = time.time()

        total_loss = 0.0
        total_f_data = 0.0
        total_f_model = 0.0

        perm = torch.randperm(len(dataset))

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            B = len(idx)

            # -----------------------------
            # Load batch
            # -----------------------------
            A_data = torch.stack([dataset[i][0] for i in idx]).to(device)
            feats = torch.stack([dataset[i][1] for i in idx]).to(device)

            A_model = torch.stack([persistent_chains[i] for i in idx]).to(device)

            # -----------------------------
            # Optional chain reset
            # -----------------------------
            reset_mask = torch.rand(B, device=device) < reset_prob
            if reset_mask.any():
                Ar = torch.bernoulli(
                    torch.full((B, N, N), init_p, device=device)
                )
                Ar = torch.triu(Ar, diagonal=1)
                A_model[reset_mask] = Ar[reset_mask] + Ar[reset_mask].transpose(1, 2)

            # -----------------------------
            # Gibbs sampling (batched)
            # -----------------------------
            A_model = gibbs_ministeps_batch(
                A_model, feats, model, device, mini_steps
            )

            # store back
            for k, i in enumerate(idx):
                persistent_chains[i] = A_model[k].cpu()

            # -----------------------------
            # PCD gradient
            # -----------------------------
            f_data = model(feats, A_data)     # [B]
            f_model = model(feats, A_model)   # [B]



            energy_reg = 1e-3 * (f_data.pow(2).mean() + f_model.pow(2).mean())

            loss = -(f_data.mean() - f_model.mean()) + energy_reg


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



def evaluate_model_batched(
    model,
    dataset,
    device,
    num_graphs=50,
    gibbs_steps=200,
    batch_size=16,
):
    """
    Batched evaluation using parallel Gibbs chains.

    Semantics:
      - Independent Gibbs chains per graph
      - Same stationary distribution as sequential version
    """
    model.eval()

    n_eval = min(num_graphs, len(dataset))
    generated = []

    with torch.no_grad():
        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)

            # -------------------------
            # Build batch
            # -------------------------
            A_batch = []
            F_batch = []
            for i in range(start, end):
                A, feats = dataset[i]
                A_batch.append(A)
                F_batch.append(feats)

            A_batch = torch.stack(A_batch, dim=0).to(device)   # [B, N, N]
            F_batch = torch.stack(F_batch, dim=0).to(device)   # [B, N, D]

            # -------------------------
            # Batched Gibbs
            # -------------------------
            A_gen_batch = gibbs_ministeps_batch(
                A_batch,
                F_batch,
                model,
                device,
                mini_steps=gibbs_steps,
            )

            # -------------------------
            # Convert to NetworkX
            # -------------------------
            for k in range(A_gen_batch.shape[0]):
                G_gen = nx.from_numpy_array(A_gen_batch[k].cpu().numpy())
                generated.append(G_gen)

    # -------------------------
    # Statistics (unchanged)
    # -------------------------
    metrics = evaluate_mmd_generated(
        dataset.graphs[:len(generated)],
        generated,
    )

    return metrics
