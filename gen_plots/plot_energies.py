import os
import pickle
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import time

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, parentdir)

from deep_ebm.stable_gnn import TGNN_EBM
from src.torch_erg.samplers import DLP_Hybrid_Sampler, GWG_Hybrid_Sampler, DLMC_Hybrid_Sampler
from deep_ebm.plotting_obs import plot_energy_comparison

from deep_ebm.utils_ebm import (
    show_graph,
    save_graph,
    compare_graphs,
    show_graph_grid,
    show_graph_grid_ba,
    compare_statistics,
    evaluate_model,
    plot_parameter_evolution,
)

from deep_ebm.plotting_obs import (
    plot_degree_ccdf,
    plot_degree_histogram,
    plot_observable_trace,
    plot_observable_bar,
)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATASET_PATH = "data/Barabasi_40_2/barabasi_albert_graphs.pkl"
#DATASET_PATH = "data/2community_30_0.3_0.08/2_comm_graphs.pkl"
#MODEL_PATH = "checkpoints/2community_30_0.3_0.08/final_model.pt"
MODEL_PATH   = "checkpoints/Barabasi_40_2/final_model.pt"
#GEN_GRAPHS_PATH = "results/hybrid_barabasi40/GWG/gwg_hybrid_sampled_graphs.pkl"
#GEN_GRAPHS_PATH = "results/hybrid_barabasi40/GWG/gwg_unconstrained_sampled_graphs.pkl"
GEN_GRAPHS_PATH = "results/hybrid_barabasi40/GWG/unconstrained_sampled_graphsMH.pkl"
#GEN_GRAPHS_PATH = "results/2community_30_0.3_0.08/unconstrained_sampled_graphs.pkl"
outdir = "results/hybrid_barabasi40/GWG"
#outdir = "results/2community_30_0.3_0.08"
os.makedirs(outdir, exist_ok=True)



NODE_FEAT_DIM = 1          # must match training
HIDDEN_DIM = 128
MP_STEPS = 4
K_HOPS_STRUCT = 3




#K_HOPS_STRUCT = 2
# -------------------------------------------------
# Load dataset
# -------------------------------------------------
with open(DATASET_PATH, "rb") as f:
    graphs_orig = pickle.load(f)

with open(GEN_GRAPHS_PATH, "rb") as f:
    graphs_gen = pickle.load(f)

print(f"Loaded {len(graphs_orig)} original graphs")
print(f"Loaded {len(graphs_gen)} generated graphs")

G_ref = graphs_orig[0]
A_ref = torch.tensor(nx.to_numpy_array(G_ref), dtype=torch.float32)

n = A_ref.shape[0]
feats = torch.ones((n, NODE_FEAT_DIM), device=device)


# -------------------------------------------------
# Load pretrained model
# -------------------------------------------------
model = TGNN_EBM(
    node_feat_dim=NODE_FEAT_DIM,
    hidden_dim=HIDDEN_DIM,
    mp_steps=MP_STEPS,
    k_hops_struct=K_HOPS_STRUCT,
    device=device,
).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Loaded pretrained EBM")

# -------------------------------------------------
# Wrap model so sampler can call f(adj)
# -------------------------------------------------
class AdjOnlyModel(nn.Module):
    def __init__(self, model, feats):
        super().__init__()
        self.model = model
        self.feats = feats

    def forward(self, adj):
        return self.model(self.feats, adj)

adj_model = AdjOnlyModel(model, feats).to(device)

plot_energy_comparison(
    pkl_path_a=GEN_GRAPHS_PATH,
    pkl_path_b=DATASET_PATH,
    model=model,                    # <- use TGNN_EBM directly
    device=device,
    node_feat_dim=NODE_FEAT_DIM,
    label_a="Generated samples",
    label_b="Training data",
    outpath=os.path.join(outdir, "ABenergy_comparison_hist_orig.pdf"),
    style="hist",                    # recommended for paper
    title= "Energy Comparison, Data vs Generated, Barabasi-Albert"
)


def compute_ba_observables_np(G):
    A = nx.to_numpy_array(G)
    deg = A.sum(axis=-1)
    edges = deg.sum() / 2.0
    deg_var = deg.var()
    triangles = np.trace(A @ A @ A) / 6.0
    return np.array([edges, deg_var, triangles])


# --- compute dataset statistics ---
obs_all = np.stack([compute_ba_observables_np(G) for G in graphs_orig])

edges_med = np.median(obs_all[:, 0])
degvar_med = np.median(obs_all[:, 1])
tri_med = np.median(obs_all[:, 2])

print("Original graph observables (medians):")
print(f"Edges: {edges_med}")
print(f"Degree variance: {degvar_med}")
print(f"Triangles: {tri_med}")

edges_mean = np.mean(obs_all[:, 0])
degvar_mean = np.mean(obs_all[:, 1])
tri_mean = np.mean(obs_all[:, 2])

print("Original graph observables (means):")
print(f"Edges: {edges_mean}")
print(f"Degree variance: {degvar_mean}")
print(f"Triangles: {tri_mean}")