import os
import random
import pickle
import torch
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

# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Experiment setup (batched PCD)
# -------------------------------------------------

# Graph
N_NODES = 40
NUM_GRAPHS = 600          
NODE_FEAT_DIM = 1
M_ATTACH = 2              

HIDDEN_DIM = 128         
MP_STEPS = 4              
K_HOPS_STRUCT = 3         

LR = 2e-5                 
WEIGHT_DECAY = 5e-5       

EPOCHS = 180              
BATCH_SIZE = 16           

PCD_MINI_STEPS = 300      
RESET_PROB = 0.06         
INIT_P = 0.09             



experiment_name = f'Barabasi_{N_NODES}_{M_ATTACH}'


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
# Dataset generation 
# -------------------------------------------------

def generate_barabasi_albert(
    num_graphs: int,
    n: int,
    m: int,
    seed: int = 0,
):
    """
    Generates scale-free graphs via preferential attachment.

    Args:
        num_graphs: number of graphs
        n: number of nodes
        m: edges added per new node (controls tail heaviness)
    """
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        s = int(rng.integers(0, 10**9))
        G = nx.barabasi_albert_graph(n=n, m=m, seed=s)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        graphs.append(G)

    return graphs


graphs = generate_barabasi_albert(
    num_graphs=NUM_GRAPHS,
    n=N_NODES,
    m=M_ATTACH,
    seed=42,
)

# -------------------------------------------------
# Save dataset
# -------------------------------------------------

DATA_DIR = os.path.join('data', experiment_name)
os.makedirs(DATA_DIR, exist_ok=True)

dataset_path = os.path.join(DATA_DIR, "barabasi_albert_graphs.pkl")
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
    gibbs_steps=300,
    batch_size=20,
)

print("Evaluation metrics:", metrics)

# -------------------------------------------------
# Grid of generated graphs
# -------------------------------------------------




generated = []
with torch.no_grad():
    for i in range(600):
        A, feats = dataset[i]
        A_gen = gibbs_ministeps(
            A,
            feats,
            model,
            device,
            mini_steps=300,
        )
        G_gen = nx.from_numpy_array(A_gen.cpu().numpy())
        generated.append(G_gen)

out_dir = f'results/{experiment_name}'
os.makedirs(out_dir, exist_ok=True)

show_graph_grid(generated, rows=2, cols=2, layout="spring", outdir=out_dir)
compare_statistics(dataset.graphs, generated)

