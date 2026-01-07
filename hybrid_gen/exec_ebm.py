import os
import random
import pickle
import torch
import networkx as nx
import numpy as np

from deep_ebm.utils_ebm import (
    show_graph,
    save_graph,
    compare_graphs,
    show_graph_grid,
    compare_statistics,
    evaluate_model,
)

from deep_ebm.gnn_ebm import (
    TGNN_EBM,
    train_one_epoch_pcd,
    gibbs_ministeps,
    train_pcd,
    train_pcd_batched,
    evaluate_model_batched
)

from deep_ebm.stable_gnn import TGNN_EBM, StableGNN_EBM

from torch.utils.data import Dataset

# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Experiment setup (batched PCD)
# -------------------------------------------------

N_NODES = 30
NUM_GRAPHS = 500

P_IN = 0.35
P_OUT = 0.05

NODE_FEAT_DIM = 1         
HIDDEN_DIM = 128
MP_STEPS = 4

LR = 2e-4
EPOCHS = 100                

# -------- PCD (batched) --------
BATCH_SIZE = 16           
PCD_MINI_STEPS = 20        
RESET_PROB = 0.05
INIT_P = 0.1


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

DATA_DIR = "data/2communitySTABLE"
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

model = StableGNN_EBM(
    node_attr_dim=NODE_FEAT_DIM,
    hidden_dim=HIDDEN_DIM,
    mp_steps=MP_STEPS,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5,
)
# -------------------------------------------------
# Training (PCD)
# -------------------------------------------------

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
    save_dir="checkpoints/2communitySTABLE",
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
save_graph(G_gen, "gen_example_2community.png")

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
    for i in range(100):
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

show_graph_grid(generated, rows=2, cols=2, layout="spring")
compare_statistics(dataset.graphs, generated)

