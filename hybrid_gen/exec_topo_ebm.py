import os
import random
import pickle
import torch
import torch
import torch.nn as nn
import networkx as nx
import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, parentdir)

from deep_ebm.utils_ebm import (
    show_graph,
    save_graph,
    compare_graphs,
    show_graph_grid,
    compare_statistics,
    evaluate_model,
    show_graph_grid,
    plot_parameter_evolution,
    show_graph_grid_sbm_2comm
)

from deep_ebm.gnn_ebm import (
    train_one_epoch_pcd,
    gibbs_ministeps,
    train_pcd,
    train_pcd_batched,
    evaluate_model_batched
)

from deep_ebm.stable_gnn import StableGNN_EBM, TGNN_EBM

from torch.utils.data import Dataset

from src.torch_erg.samplers import DLP_Hybrid_Sampler, GWG_Hybrid_Sampler, DLMC_Hybrid_Sampler

# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Experiment setup (batched PCD)
# -------------------------------------------------


N_NODES = 30
P_IN = 0.36
P_OUT = 0.02
NUM_GRAPHS = 500
NODE_FEAT_DIM = 1

HIDDEN_DIM = 128
MP_STEPS = 4
K_HOPS_STRUCT = 3

LR = 3e-5
WEIGHT_DECAY = 1e-4

EPOCHS = 200
BATCH_SIZE = 16

PCD_MINI_STEPS = 200
RESET_PROB = 0.08
INIT_P = 0.12

experiment_name = f'2community_{N_NODES}_{P_IN}_{P_OUT}'


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class GraphDataset(Dataset):
    """
    Permutation-invariant dataset:
      - adjacency
      - constant node features
    """
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        G = self.graphs[idx]
        n = G.number_of_nodes()

        A = nx.to_numpy_array(G).astype(np.float32)
        feats = np.ones((n, NODE_FEAT_DIM), dtype=np.float32)

        return torch.from_numpy(A), torch.from_numpy(feats)

# -------------------------------------------------
# Dataset generation (2-community SBM)
# -------------------------------------------------

def generate_2community_sbm(
    num_graphs: int,
    n: int,
    p_in: float,
    p_out: float,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    graphs = []

    sizes = [n // 2, n - n // 2]
    probs = [[p_in, p_out], [p_out, p_in]]

    for _ in range(num_graphs):
        s = int(rng.integers(0, 10**9))
        G = nx.stochastic_block_model(sizes, probs, seed=s)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        graphs.append(G)

    return graphs


graphs = generate_2community_sbm(
    num_graphs=NUM_GRAPHS,
    n=N_NODES,
    p_in=P_IN,
    p_out=P_OUT,
    seed=42,
)

# -------------------------------------------------
# Save dataset
# -------------------------------------------------

DATA_DIR = os.path.join('data', experiment_name)
os.makedirs(DATA_DIR, exist_ok=True)

dataset_path = os.path.join(DATA_DIR, "2_comm_graphs.pkl")
with open(dataset_path, "wb") as f:
    pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[INFO] Saved {len(graphs)} graphs to {dataset_path}")

# -------------------------------------------------
# Dataset wrapper
# -------------------------------------------------

dataset = GraphDataset(graphs)

# -------------------------------------------------
# Model
# -------------------------------------------------

model = TGNN_EBM(
    hidden_dim=HIDDEN_DIM,
    mp_steps=MP_STEPS,
    k_hops_struct=K_HOPS_STRUCT,
).to(device)

LR = 2e-5          # or 1e-5 if still unstable
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    betas=(0.5, 0.9),
    weight_decay=1e-4,   # stronger than 1e-5
)
# -------------------------------------------------
# Training (PCD)
# -------------------------------------------------

save_dir = os.path.join('checkpoints', experiment_name)
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
A_gen = gibbs_ministeps(
    A0,
    feats0,
    model,
    device,
    mini_steps=200,
)

G_real = nx.from_numpy_array(A0.numpy())
G_gen = nx.from_numpy_array(A_gen.cpu().numpy())

compare_graphs(G_real, G_gen)
save_graph(G_gen, f"gen_example_{experiment_name}.png")

# -------------------------------------------------
# Quantitative evaluation
# -------------------------------------------------

metrics = evaluate_model_batched(
    model,
    dataset,
    device,
    num_graphs=20,
    gibbs_steps=1000,
    batch_size=20,
)

print("Evaluation metrics:", metrics)

# -------------------------------------------------
# Grid of generated graphs
# -------------------------------------------------

generated = []

with torch.no_grad():
    for i in range(25):   # fewer chains
        n = dataset[0][0].shape[0]
        feats = torch.ones((n, NODE_FEAT_DIM), device=device)

        # random init
        A = (torch.rand(n, n, device=device) < INIT_P).float()
        A = torch.triu(A, diagonal=1)
        A = A + A.T

        # burn-in
        A = gibbs_ministeps(A, feats, model, device, mini_steps=1000)

        # collect multiple samples
        for _ in range(4):
            A = gibbs_ministeps(A, feats, model, device, mini_steps=100)
            generated.append(nx.from_numpy_array(A.cpu().numpy()))

out_dir = f'results/{experiment_name}'
os.makedirs(out_dir, exist_ok=True)

try:
    with open(os.path.join(out_dir, 'unconstrained_sampled_graphs.pkl'), 'wb') as f:
        pickle.dump(generated, f)
    print(f"Saved sampled graphs to {os.path.join(out_dir, 'unconstrained_sampled_graphs.pkl')}")
except Exception as e:
    print(f"Failed to save sampled graphs: {e}")

show_graph_grid(generated, rows=2, cols=2, layout="spring", outdir=out_dir)
compare_statistics(dataset.graphs, generated)

try:
    show_graph_grid_sbm_2comm(
        generated,
        n1=N_NODES // 2,
        rows=2,
        cols=2,
        outdir=out_dir,
        filename='unconstrained_generation_sbm_layout.png'
    )
except Exception as e:
    print(f"Failed to create SBM layout grid: {e}")


DATASET_PATH = dataset_path
MODEL_PATH = os.path.join(save_dir, "final_model.pt")

outdir = os.path.join('results', experiment_name, 'gwg_hybrid')
os.makedirs(outdir, exist_ok=True)



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

edges_q30 = np.quantile(obs_all[:, 0], 0.3)
tri_q30   = np.quantile(obs_all[:, 1], 0.3)

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
    niter=3000000,
    params_update_every=5,
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

try:
    show_graph_grid_sbm_2comm(
        selected_graphs[-4:],
        n1=N_NODES // 2,
        rows=2,
        cols=2,
        outdir=outdir,
        filename='gwg_hybrid_sbm_layout.png'
    )
except Exception as e:
    print(f"Failed to create SBM layout grid: {e}") 