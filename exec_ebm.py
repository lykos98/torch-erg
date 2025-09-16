import random
import networkx as nx
import torch

from deep_ebm.utils_ebm import show_graph, evaluate_model
from deep_ebm.utils_ebm import save_graph, compare_graphs, show_graph_grid, compare_statistics
from deep_ebm.gnn_ebm import GraphDataset, GNN_EBM, train_one_epoch_pcd, gibbs_ministeps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# make toy dataset of small ER graphs
graphs = []
for _ in range(200):
    n = random.randint(5, 15)
    G = nx.erdos_renyi_graph(n, 0.2)
    graphs.append(G)

dataset = GraphDataset(graphs)
model = GNN_EBM(node_feat_dim=dataset.max_nodes, hidden_dim=64, mp_steps=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

persistent = []
for epoch in range(50):
    loss = train_one_epoch_pcd(model, dataset, opt, device, persistent_chains=persistent, mini_steps=20)
    print(f"Epoch {epoch} loss = {loss:.4f}")


A, feats = dataset[0]
A_gen = gibbs_ministeps(A, model, feats.to(device), device, mini_steps=1000)
G_real = nx.from_numpy_array(A.numpy())
G_gen = nx.from_numpy_array(A_gen.cpu().numpy())

# Show side-by-side
compare_graphs(G_real, G_gen)

# Or save to file
save_graph(G_gen, "gen_example.png")

# Evaluate over multiple graphs
metrics = evaluate_model(model, dataset, device, num_graphs=10, gibbs_steps=200)
print("MMD metrics:", metrics)  

# Show grid of generated graphs
generated = []
with torch.no_grad():
    for i in range(81):
        A, feats = dataset[i]
        A_gen = gibbs_ministeps(A, model, feats.to(device), device, mini_steps=200)
        G_gen = nx.from_numpy_array(A_gen.cpu().numpy())
        generated.append(G_gen) 

show_graph_grid(generated, rows=9, cols=9, layout="spring")

compare_statistics(dataset.graphs, generated)


