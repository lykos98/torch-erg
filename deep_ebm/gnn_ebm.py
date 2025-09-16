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

# -----------------------------
# Model components
# -----------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMP(nn.Module):
    """Single message-passing step using dot-product attention.

    Given node features H (n x d), compute messages via attention between nodes
    and update node features with an MLP.
    """
    def __init__(self, in_dim, out_dim, device: torch.device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # Linear projections for query/key/value
        self.q = nn.Linear(in_dim, out_dim, bias=False).to(device)
        self.k = nn.Linear(in_dim, out_dim, bias=False).to(device)
        self.v = nn.Linear(in_dim, out_dim, bias=False).to(device)

        # MLP update
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        ).to(device)

    def forward(self, H: torch.Tensor, adj: torch.Tensor):
        H = H.to(self.device)
        adj = adj.to(self.device)

        q = self.q(H)
        k = self.k(H)
        v = self.v(H)

        scores = torch.matmul(q, k.t())  # n x n
        att_mask = (adj > 0).float()
        masked_scores = scores - (1.0 - att_mask) * 1e6
        att = F.softmax(masked_scores, dim=-1)

        M = torch.matmul(att, v)
        out = self.mlp(M)
        return out


class GNN_EBM(nn.Module):
    def __init__(self, node_feat_dim: int = 16, hidden_dim: int = 1024, mp_steps: int = 6, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        self.node_in = nn.Linear(node_feat_dim, hidden_dim).to(device)
        self.mps = nn.ModuleList([AttentionMP(hidden_dim, hidden_dim, device) for _ in range(mp_steps)])

        # readout: mean pooling then MLP to scalar
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        node_feats = node_feats.to(self.device)
        adj = adj.to(self.device)

        h = self.node_in(node_feats)
        for mp in self.mps:
            delta = mp(h, adj)
            h = h + delta  # residual

        g = h.mean(dim=0)  # mean pooling
        out = self.readout(g)
        return out.squeeze()


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
        feats = np.zeros((n, self.max_nodes), dtype=np.float32)
        for i in range(n):
            feats[i, i] = 1.0
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
    #compute maximum mean discrepancy between two sets of samples
    # using RBF kernel with bandwidth sigma
    # X, Y are arrays of shape (n_samples, d) or lists of vectors; we'll compute pairwise RBF kernel MMD
    X = np.asarray(X)
    Y = np.asarray(Y)
    def rbf(a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        diff = a - b
        return math.exp(-np.dot(diff, diff) / (2 * sigma * sigma))
    m = len(X)
    n = len(Y)
    xx = sum(rbf(X[i], X[j]) for i in range(m) for j in range(m)) / (m * m)
    yy = sum(rbf(Y[i], Y[j]) for i in range(n) for j in range(n)) / (n * n)
    xy = sum(rbf(X[i], Y[j]) for i in range(m) for j in range(n)) / (m * n)
    return xx + yy - 2 * xy


def evaluate_mmd_generated(dataset_graphs: List[nx.Graph], generated_graphs: List[nx.Graph], sigma=1.0):
    # degree MMD (maximum mean discrepancy)
    deg_real = [graph_degree_hist(G) for G in dataset_graphs]
    deg_gen = [graph_degree_hist(G) for G in generated_graphs]
    # pad to equal length vectors for kernel: pad with zeros to max nodes
    maxlen = max(max(len(v) for v in deg_real), max(len(v) for v in deg_gen))
    def pad(v):
        return np.pad(v, (0, maxlen - len(v)))
    deg_real = [pad(v) for v in deg_real]
    deg_gen = [pad(v) for v in deg_gen]
    mmd_deg = compute_mmd(np.array(deg_real), np.array(deg_gen), sigma=sigma)

    # clustering MMD 
    clus_real = [graph_clustering_vec(G) for G in dataset_graphs]
    clus_gen = [graph_clustering_vec(G) for G in generated_graphs]
    maxlen = max(max(len(v) for v in clus_real), max(len(v) for v in clus_gen))
    clus_real = [pad(v) for v in clus_real]
    clus_gen = [pad(v) for v in clus_gen]
    mmd_clus = compute_mmd(np.array(clus_real), np.array(clus_gen), sigma=sigma)

    

    return {"degree_mmd": mmd_deg, "clustering_mmd": mmd_clus}




# -----------------------------
# Training harness
# -----------------------------

def train_one_epoch_pseudolikelihood(model: nn.Module, dataset: GraphDataset, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    for A, feats in dataset:
        feats = feats.to(device)
        A = A.to(device)
        optimizer.zero_grad()
        loss = pseudolikelihood_loss(model, A, feats, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


def train_one_epoch_cnce(model: nn.Module, dataset: GraphDataset, optimizer: torch.optim.Optimizer, device: torch.device, alpha=1.0, beta=20.0):
    model.train()
    total_loss = 0.0
    for A, feats in dataset:
        feats = feats.to(device)
        A = A.to(device)
        optimizer.zero_grad()
        loss = cnce_loss(model, A, feats, lambda x: cnce_corrupt(x, alpha, beta), device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


def train_one_epoch_pcd(model: nn.Module, dataset: GraphDataset, optimizer: torch.optim.Optimizer, device: torch.device, persistent_chains: List[torch.Tensor], mini_steps:int = 100):
    """persistent_chains: list of adjacency matrices same length or greater than dataset; if empty, initialize from ER(p=0.1)"""
    model.train()
    total_loss = 0.0
    # ensure persistent_chains length >= dataset
    if len(persistent_chains) < len(dataset):
        # initialize additional chains from Erdos-Renyi p=0.1 with corresponding sizes
        for A, _ in dataset:
            n = A.shape[0]
            if len(persistent_chains) >= len(dataset):
                break
            p = 0.1
            Ar = torch.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        Ar[i, j] = 1.0
                        Ar[j, i] = 1.0
            persistent_chains.append(Ar)
    # train
    idx = 0
    for A, feats in dataset:
        feats = feats.to(device)
        A = A.to(device)
        # get persistent chain for this graph
        chain_A = persistent_chains[idx]
        if chain_A.shape[0] != A.shape[0]:
            # re-init chain for this size
            n = A.shape[0]
            p = 0.1
            Ar = torch.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        Ar[i, j] = 1.0
                        Ar[j, i] = 1.0
            chain_A = Ar
            persistent_chains[idx] = chain_A
        # run mini-steps to update chain
        chain_A = gibbs_ministeps(chain_A, model, feats, device, mini_steps)
        persistent_chains[idx] = chain_A
        optimizer.zero_grad()
        loss = pcd_loss(model, A, feats, chain_A.to(device), device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        idx += 1
    return total_loss / len(dataset)

# -----------------------------
# Generation utilities
# -----------------------------

def greedy_minimize_energy(adj: torch.Tensor, model: nn.Module, feats: torch.Tensor, device: torch.device, max_steps=100) -> torch.Tensor:
    A = adj.clone()
    n = A.shape[0]
    for _ in range(max_steps):
        improved = False
        best_delta = 0.0
        best_pair = None
        fA = model(feats.to(device), A.to(device)).item()
        for i in range(n):
            for j in range(i+1, n):
                A_prime = A.clone()
                A_prime[i,j] = 1.0 - A_prime[i,j]
                A_prime[j,i] = A_prime[i,j]
                fApr = model(feats.to(device), A_prime.to(device)).item()
                if fApr > fA + 1e-8:
                    delta = fApr - fA
                    if delta > best_delta:
                        best_delta = delta
                        best_pair = (i,j)
        if best_pair is None:
            break
        # apply best flip
        i,j = best_pair
        A[i,j] = 1.0 - A[i,j]
        A[j,i] = A[i,j]
    return A


