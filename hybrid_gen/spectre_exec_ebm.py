import os
import random
import torch
import networkx as nx
import numpy as np

from torch.utils.data import Dataset

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, parentdir)

from deep_ebm.utils_ebm import (
    save_graph,
    compare_graphs,
    show_graph_grid,
    compare_statistics,
)

from deep_ebm.gnn_ebm import (
    gibbs_ministeps,
    train_pcd_batched,
    evaluate_model_batched,
)

from deep_ebm.stable_gnn import TGNN_EBM




# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Experiment setup
# -------------------------------------------------
NUM_GRAPHS = 600
NODE_FEAT_DIM = 1

HIDDEN_DIM = 128
MP_STEPS = 4
K_HOPS_STRUCT = 3

LR = 2e-5
WEIGHT_DECAY = 1e-4

EPOCHS = 180
BATCH_SIZE = 16

PCD_MINI_STEPS = 300
RESET_PROB = 0.06
INIT_P = 0.09

DATASET_NAME = "planar_64_200"
experiment_name = DATASET_NAME


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def sanitize_adj(A: torch.Tensor) -> torch.Tensor:
    """Make A symmetric, zero diagonal, binary in {0,1}."""
    A = A.float()
    A = A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))  # zero diag
    A = 0.5 * (A + A.transpose(-1, -2))                            # symmetrize
    A = (A > 0.5).float()                                          # binarize
    return A


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class SpectreAdjDataset(Dataset):
    def __init__(self, pt_path, node_feat_dim=1, keep_only_full_size=True):
        payload = torch.load(pt_path, map_location="cpu")
        self.adjs, self.eigvals, self.eigvecs, self.n_nodes, *_ = payload

        assert isinstance(self.adjs, list)

        self.adjs = [A.float() for A in self.adjs]
        self.n_nodes_vec = torch.tensor([A.size(0) for A in self.adjs])
        self.n_max = int(self.n_nodes_vec.max().item())
        self.node_feat_dim = node_feat_dim

        if keep_only_full_size:
            self.keep_idx = (self.n_nodes_vec == self.n_max).nonzero(as_tuple=False).view(-1)
        else:
            self.keep_idx = None

    def __len__(self):
        return len(self.keep_idx) if self.keep_idx is not None else len(self.adjs)

    def __getitem__(self, i):
        idx = self.keep_idx[i].item() if self.keep_idx is not None else i
        A = self.adjs[idx]

        # sanitize
        A = A.clone()
        A.fill_diagonal_(0.0)
        A = 0.5 * (A + A.t())
        A = (A > 0.5).float()

        n = A.size(0)
        feats = torch.ones((n, self.node_feat_dim), dtype=torch.float32)

        return A, feats



# -------------------------------------------------
# Data path (HPC-safe)
# -------------------------------------------------
SPECTRE_DATA_DIR = os.environ.get(
    "SPECTRE_DATA",
    "/leonardo_scratch/large/userexternal/fgiacoma/SPECTRE", #cambiare con percorso appropriato
)
if not os.path.isdir(SPECTRE_DATA_DIR):
    raise FileNotFoundError(f"SPECTRE_DATA directory not found: {SPECTRE_DATA_DIR}")

pt_path = os.path.join(SPECTRE_DATA_DIR, f"{DATASET_NAME}.pt")
dataset = SpectreAdjDataset(pt_path, node_feat_dim=NODE_FEAT_DIM, keep_only_full_size=True)

N_NODES = dataset.n_max
print(f"[INFO] Loaded SPECTRE dataset '{DATASET_NAME}' with N={N_NODES}, num_graphs={len(dataset)}")
print(f"[INFO] Using data path: {pt_path}")


# -------------------------------------------------
# Model
# -------------------------------------------------
model = TGNN_EBM(
    node_feat_dim=NODE_FEAT_DIM,
    hidden_dim=HIDDEN_DIM,
    mp_steps=MP_STEPS,
    k_hops_struct=K_HOPS_STRUCT,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    betas=(0.5, 0.9),
    weight_decay=WEIGHT_DECAY,
)


# -------------------------------------------------
# Training (PCD)
# -------------------------------------------------
save_dir = os.path.join("checkpoints", experiment_name)
os.makedirs(save_dir, exist_ok=True)

persistent_chains = train_pcd_batched(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    device=device,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    mini_steps=PCD_MINI_STEPS,
    reset_prob=RESET_PROB,
    init_p=INIT_P,
    print_every=1,
    save_dir=save_dir,
)


# -------------------------------------------------
# Single-sample generation demo
# -------------------------------------------------
A0, feats0 = dataset[0]
A_gen = gibbs_ministeps(A0, feats0, model, device, mini_steps=200)

G_real = nx.from_numpy_array(A0.numpy())
G_gen = nx.from_numpy_array(A_gen.cpu().numpy())

compare_graphs(G_real, G_gen)

out_dir = os.path.join("results", experiment_name)
os.makedirs(out_dir, exist_ok=True)
save_graph(G_gen, os.path.join(out_dir, f"gen_example_{experiment_name}.png"))


# -------------------------------------------------
# Quantitative evaluation
# -------------------------------------------------
metrics = evaluate_model_batched(
    model,
    dataset,
    device,
    num_graphs=min(20, len(dataset)),
    gibbs_steps=300,
    batch_size=min(20, len(dataset)),
)
print("Evaluation metrics:", metrics)


# -------------------------------------------------
# Grid of generated graphs + statistics
# -------------------------------------------------
n_show = min(NUM_GRAPHS, len(dataset))
generated = []
real_graphs = []

with torch.no_grad():
    for i in range(n_show):
        A, feats = dataset[i]
        real_graphs.append(nx.from_numpy_array(A.numpy()))

        A_gen = gibbs_ministeps(A, feats, model, device, mini_steps=300)
        generated.append(nx.from_numpy_array(A_gen.cpu().numpy()))

# Save a small grid
show_graph_grid(generated[:4], rows=2, cols=2, layout="spring", outdir=out_dir)

# Compare stats: REAL vs GENERATED
compare_statistics(real_graphs, generated)
