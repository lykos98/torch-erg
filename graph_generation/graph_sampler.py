import os
import pickle
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import time
#from deep_ebm.stable_gnn import TGNN_EBM

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, parentdir)

from deep_ebm.gnn_ebm import TGNN_EBM
from src.torch_erg.samplers import DLP_Hybrid_Sampler, GWG_Hybrid_Sampler, DLMC_Hybrid_Sampler

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

from deep_ebm.gnn_ebm import (
    train_one_epoch_pcd,
    gibbs_ministeps,
    train_pcd,
    train_pcd_batched,
    evaluate_model_batched
)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Paths
# -------------------------------------------------
# DATASET_PATH = "data/Barabasi_40_2/barabasi_albert_graphs.pkl"
# MODEL_PATH   = "checkpoints/Barabasi_40_2/final_model.pt"
# outdir = "results/hybrid_barabasi40/GWG"

# DATASET_PATH = "data/2community_30_0.3_0.08/2_comm_graphs.pkl"
# MODEL_PATH   = "checkpoints/2community_30_0.3_0.08/final_model.pt"
# outdir = "results/2community/sampler"

DATASET_PATH = "data/2community/2_comm_graphs.pkl"
HYBRID_PATH =  "results/hybrid_2comm/GWG/gwg_hybrid_sampled_graphs.pkl"
MODEL_PATH   = "checkpoints/2community/final_model.pt"
outdir = "results/hybrid_2comm/sampler"
os.makedirs(outdir, exist_ok=True)



N_NODES= 30
NODE_FEAT_DIM = 1          # must match training
HIDDEN_DIM = 128
MP_STEPS = 4
K_HOPS_STRUCT = 3



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

generated = []

BURN_IN = 2000
THINNING = 100
N_CHAINS = 20
SAMPLES_PER_CHAIN = 5
INIT_P = 0.12

generated = []

with torch.no_grad():
    for _ in range(N_CHAINS):
        n = N_NODES  # use your known N=80, don't read from dataset here
        feats = torch.ones((n, NODE_FEAT_DIM), device=device)

        # random init (use a density close to BA expected density)
        # expected avg degree ≈ 2m  -> p ≈ (2m)/(n-1)
        p0 = INIT_P
        A = (torch.rand(n, n, device=device) < p0).float()
        A = torch.triu(A, diagonal=1)
        A = A + A.T

        # burn-in
        A = gibbs_ministeps(A, feats, model, device, mini_steps=BURN_IN)

        # collect thinned samples
        for _ in range(SAMPLES_PER_CHAIN):
            A = gibbs_ministeps(A, feats, model, device, mini_steps=THINNING)
            generated.append(nx.from_numpy_array(A.cpu().numpy()))



try:
    out_path_file = os.path.join(outdir, 'unconstrained_sampled_graphsMH.pkl')
    os.makedirs(outdir, exist_ok=True)
    
    with open(out_path_file, 'wb') as f:
        pickle.dump(generated, f)
    print(f"Saved sampled graphs to {out_path_file}")
except Exception as e:
    print(f"Failed to save sampled graphs: {e}")


def sbm_observables_np(G):
    A = nx.to_numpy_array(G)
    edges = A.sum() / 2.0
    triangles = np.trace(A @ A @ A) / 6.0
    return np.array([edges, triangles])

# -------------------------------------------------
# Target observables = 80% of observed
# -------------------------------------------------
obs_all = np.stack([sbm_observables_np(G) for G in generated])

edges_mean = np.mean(obs_all[:, 0])
tri_mean   = np.mean(obs_all[:, 1])

edges_median = np.median(obs_all[:, 0])
tri_median   = np.median(obs_all[:, 1])
print(f"Generated mean observables: edges={edges_mean}, triangles={tri_mean}")
print(f"Generated median observables: edges={edges_median}, triangles={tri_median}")

#original graphs for comparison
with open(DATASET_PATH, "rb") as f:
    graphs_orig = pickle.load(f)    

obs_all_orig = np.stack([sbm_observables_np(G) for G in graphs_orig])
edges_mean_data = np.mean(obs_all_orig[:, 0])
tri_mean_data   = np.mean(obs_all_orig[:, 1])
edges_median_data = np.median(obs_all_orig[:, 0])
tri_median_data   = np.median(obs_all_orig[:, 1])

print(f"Dataset mean observables: edges={edges_mean_data}, triangles={tri_mean_data}")
print(f"Dataset median observables: edges={edges_median_data}, triangles={tri_median_data}")

#sampled graph with method hybrid GWG
with open(HYBRID_PATH, "rb") as f:
    graphs_hybrid = pickle.load(f)
obs_all_hybrid = np.stack([sbm_observables_np(G) for G in graphs_hybrid])
edges_mean_hybrid = np.mean(obs_all_hybrid[:, 0])
tri_mean_hybrid   = np.mean(obs_all_hybrid[:, 1])
edges_median_hybrid = np.median(obs_all_hybrid[:, 0])
tri_median_hybrid   = np.median(obs_all_hybrid[:, 1])

print(f"Hybrid GWG sampled mean observables: edges={edges_mean_hybrid}, triangles={tri_mean_hybrid}")
print(f"Hybrid GWG sampled median observables: edges={edges_median_hybrid}, triangles={tri_median_hybrid}")

selected_hybrid = graphs_hybrid[:10]
show_graph_grid(selected_hybrid[-4:], rows=2, cols=2, layout="spring",outdir=outdir,filename='hybrid_GWG_generation')

show_graph_grid(graphs_orig[-4:], rows=2, cols=2, outdir=outdir,filename='original_graphs')

# --- compute dataset statistics ---

selected_graphs = generated[:10]
show_graph_grid(selected_graphs[-4:], rows=2, cols=2, layout="spring",outdir=outdir,filename='unconstrained_generation')


def plot_sbm_separated_layout(G, n1=15, seed=0, outdir=None, filename=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    # initial positions: two clusters left/right
    init_pos = {}
    for i in range(G.number_of_nodes()):
        if i < n1:
            init_pos[i] = np.array([-1.0, np.random.randn()*0.1])
        else:
            init_pos[i] = np.array([+1.0, np.random.randn()*0.1])

    pos = nx.spring_layout(G, seed=seed, pos=init_pos, fixed=None, iterations=200)
    colors = ["tab:blue" if i < 100 else "tab:orange" for i in G.nodes()]
    nx.draw(G, pos, node_color=colors, node_size=120, width=0.8, alpha=0.9)
    plt.savefig(os.path.join(outdir, f"{filename}.png"), dpi=200)
    plt.close()

for i, G in enumerate(selected_graphs[-4:]):
    plot_sbm_separated_layout(G, n1=15, seed=i, outdir=outdir, filename=f"unconstrained_generation_{i}")

for i, G in enumerate(graphs_orig[:4]):
    plot_sbm_separated_layout(G, n1=15, seed=i, outdir=outdir, filename=f"original_graph_{i}")

for i, G in enumerate(selected_hybrid[:4]):
    plot_sbm_separated_layout(G, n1=15, seed=i, outdir=outdir, filename=f"hybrid_GWG_generation_{i}")