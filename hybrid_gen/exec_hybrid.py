import os
import pickle
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import time
#from deep_ebm.gnn_ebm import TGNN_EBM

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, parentdir)

from deep_ebm.stable_gnn import TGNN_EBM
from src.torch_erg.samplers import DLP_Hybrid_Sampler, GWG_Hybrid_Sampler, DLMC_Hybrid_Sampler

from deep_ebm.utils_ebm import (
    show_graph,
    save_graph,
    compare_graphs,
    show_graph_grid,
    compare_statistics,
    evaluate_model,
    plot_parameter_evolution,
)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATASET_PATH = "data/2community_30_0.3_0.08/2_comm_graphs.pkl"
MODEL_PATH = "checkpoints/2community_30_0.3_0.08/final_model.pt"

outdir = "results/hybrid_2comm30_0.3_0.08/GWG"
os.makedirs(outdir, exist_ok=True)

NODE_FEAT_DIM = 1          # must match training
HIDDEN_DIM = 128
MP_STEPS = 4
K_HOPS_STRUCT = 2


# -------------------------------------------------
# Load dataset
# -------------------------------------------------
with open(DATASET_PATH, "rb") as f:
    graphs = pickle.load(f)

print(f"Loaded {len(graphs)} graphs")


show_graph_grid(graphs[-4:], rows=2, cols=2, layout="spring",outdir=outdir,filename='original_graphs')
# Use first graph as reference
G_ref = graphs[0]
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

# -------------------------------------------------
# Define observables
# -------------------------------------------------
def edge_and_triangle_counts(adj: torch.Tensor):
    """
    adj: (n,n) binary adjacency
    returns: tensor([edges, triangles])
    """
    edges = torch.sum(adj) / 2.0
    triangles = torch.trace(adj @ adj @ adj) / 6.0
    return torch.stack([edges, triangles])

def sbm_observables_np(G):
    A = nx.to_numpy_array(G)
    edges = A.sum() / 2.0
    triangles = np.trace(A @ A @ A) / 6.0
    return np.array([edges, triangles])

# -------------------------------------------------
# Compute observed observables
# -------------------------------------------------
A_ref = A_ref.to(device)
obs = edge_and_triangle_counts(A_ref)

print("Observed observables:")
print("  edges     =", obs[0].item())
print("  triangles =", obs[1].item())

# -------------------------------------------------
# Target observables = 80% of observed
# -------------------------------------------------
obs_all = np.stack([sbm_observables_np(G) for G in graphs])

edges_mean = np.mean(obs_all[:, 0])
tri_mean   = np.mean(obs_all[:, 1])

edges_q30 = 73
tri_q30   = 34

target_obs = torch.tensor(
    [edges_q30, tri_q30],
    device=device,
    dtype=torch.float32,
)

print("All graph observables (mean):")
print("  edges     =", edges_mean)
print("  triangles =", tri_mean)    

print("Target observables (70%):")
print("  edges     =", target_obs[0].item())
print("  triangles =", target_obs[1].item())

# -------------------------------------------------
# EE / GWG hybrid sampler
# -------------------------------------------------
class MySampler(GWG_Hybrid_Sampler):
    def __init__(self, backend: str, model: nn.Module, mod_ratio: bool = False):
        super().__init__(backend, model, mod_ratio)
        

    def observables(self, adj):
        return edge_and_triangle_counts(adj)

sampler = MySampler(
    backend="cuda" if device.type == "cuda" else "cpu",
    model=adj_model, mod_ratio=False
)

# Initial beta parameters
betas = torch.zeros(2, device=device)

# -------------------------------------------------
# Run EE parameter estimation
# -------------------------------------------------


start_graph = torch.zeros_like(A_ref)
start_time_params = time.time()
parlist, ret_graphs = sampler.param_run(
    graph=A_ref,
    observables=target_obs,
    params=betas,
    niter=1000000,
    params_update_every=3,
    save_every=50,
    save_params=True,
    alpha=5e-5,
    min_change=2e-3,
)
end_time_params = time.time()
print(f"Parameter estimation took {end_time_params - start_time_params:.2f} seconds")


plot_parameter_evolution(
    parlist_np=torch.stack(parlist).cpu().numpy(),
    outdir=outdir,
    filename="parameter_evolution.png",
    title="Parameter Evolution - GWG Hybrid on 2-community",
    burnin=None,
    dpi=200,
)
params_est = torch.stack(parlist[-30:]).mean(dim=0)


print("Estimated beta parameters:")
print(params_est)

start_time_sampling = time.time()
obs_list,graphs_list = sampler.sample_run(graph=A_ref,
                      params=params_est,
                      niter=200000,
                      save_every=50,
                      burn_in = 0.3
                      )


end_time_sampling = time.time()
print(f"Sampling took {end_time_sampling - start_time_sampling:.2f} seconds")

obs_list0,graphs_list0 = sampler.sample_run(graph=A_ref,
                      params=torch.zeros_like(params_est),
                      niter=200000,
                      save_every=50,
                      burn_in = 0.3
                      )

graphs_nx = [
    nx.from_numpy_array(A.cpu().numpy())
    for A in graphs_list
]

graphs_nx0 = [
    nx.from_numpy_array(A.cpu().numpy())
    for A in graphs_list0
]

selected_graphs = graphs_nx[::10]  # select every 10th graph to reduce number

selected_graphs0 = graphs_nx0[::10]  # select every 10th graph to reduce number

try:
    with open(os.path.join(outdir, 'gwg_hybrid_sampled_graphs.pkl'), 'wb') as f:
        pickle.dump(selected_graphs, f)
    print(f"Saved sampled graphs to {os.path.join(outdir, 'gwg_hybrid_sampled_graphs.pkl')}")
except Exception as e:
    print(f"Failed to save sampled graphs: {e}")


try:
    with open(os.path.join(outdir, 'gwg_unconstrained_sampled_graphs.pkl'), 'wb') as f:
        pickle.dump(selected_graphs0, f)
    print(f"Saved sampled graphs to {os.path.join(outdir, 'gwg_unconstrained_sampled_graphs.pkl')}")
except Exception as e:
    print(f"Failed to save sampled graphs: {e}")

show_graph_grid(selected_graphs[-4:], rows=2, cols=2, layout="spring",outdir=outdir,filename='gwg_hybrid')
show_graph_grid(selected_graphs0[-4:], rows=2, cols=2, layout="spring",outdir=outdir,filename='gwg_no_constraint')
compare_statistics(graphs, selected_graphs)
compare_statistics(graphs, selected_graphs0)
