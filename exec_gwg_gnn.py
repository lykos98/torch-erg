import random
import networkx as nx
import torch
import torch.nn as nn

from deep_ebm.utils_ebm import show_graph, evaluate_model
from deep_ebm.utils_ebm import save_graph, compare_graphs, show_graph_grid, compare_statistics
from deep_ebm.gnn_ebm import GraphDataset, GNN_EBM, train_one_epoch_pcd, gibbs_ministeps

from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.samplers import GWGSampler, MHSampler, GWG_Hybrid_Sampler
from src.torch_erg.utils import laplacian_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# change to synthetic 30_ieee dataset
graphs = []
for _ in range(200):
    n = 30
    G = nx.erdos_renyi_graph(n, 0.2)
    graphs.append(G)

dataset = GraphDataset(graphs)
Gmodel = GNN_EBM(node_feat_dim=dataset.max_nodes, hidden_dim=64, mp_steps=2, device=device).to(device)
opt = torch.optim.Adam(Gmodel.parameters(), lr=1e-4)

persistent = []
for epoch in range(50):
    loss = train_one_epoch_pcd(Gmodel, dataset, opt, device, persistent_chains=persistent, mini_steps=10)
    print(f"Epoch {epoch} loss = {loss:.4f}")


#IMPORTANT! Here we are considering the node feats of a generic element of the dataset as all the elements have the same feats
A, feats = dataset[0]

print('node feats are: ', feats)
A_gen = gibbs_ministeps(A, Gmodel, feats.to(device), device, mini_steps=2000)
G_real = nx.from_numpy_array(A.numpy())
G_gen = nx.from_numpy_array(A_gen.cpu().numpy())

# Show side-by-side
compare_graphs(G_real, G_gen)

# Or save to file
save_graph(G_gen, "gen_example.png")

# Evaluate over multiple graphs
metrics = evaluate_model(Gmodel, dataset, device, num_graphs=10, gibbs_steps=200)
print("MMD metrics:", metrics)  

#check se grafi generati dal modello sono connessi

#decidere osservabili target e verificare valori prima (modello deep vanilla) e dopo GWG (con parametri stimati)

# Show grid of generated graphs
generated = []
with torch.no_grad():
    for i in range(81):
        A, feats = dataset[i]
        A_gen = gibbs_ministeps(A, Gmodel, feats.to(device), device, mini_steps=200)
        G_gen = nx.from_numpy_array(A_gen.cpu().numpy())
        generated.append(G_gen) 

        
compare_statistics(dataset.graphs, generated)




name = '30_ieee'

# soft connectivity constraint flag, if false then hard constraint is used
SOFT_CONN = True

ordmat, ordlist, buslist, countlist = lp.pow_parser(name)

ordmat = torch.tensor(ordmat)
ordlist = torch.tensor(ordlist)
buslist = torch.tensor(buslist)
countlist = torch.tensor(countlist)

G_sparse = ordmat.cpu().numpy()  # Create sparse matrix
n_components = connected_components(csr_matrix(G_sparse))
print("Number of connected components in the graph:", n_components[0])

class MySampler(GWG_Hybrid_Sampler):
    def __init__(self, backend: str, model: nn.Module):
        super().__init__(backend, model)
    def observables(self, mtx):
        edges = torch.sum(mtx)/2
        triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
        #ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
        return(torch.stack([edges, triangles]))
        
betas = torch.tensor([0., 0.], dtype=float)

Gmodel.eval().to(device)



#we freeze the node features as we do not need them, so that we can evaluate the model only on the adj matrix
class feat_encoded_model(nn.Module):
    def __init__(self, model, feats):
        super().__init__()
        self.model = model
        self.feats = feats.to(device)

    def forward(self,adj):
        return(self.model(self.feats, adj))

adj_model = feat_encoded_model(Gmodel, feats)
sampler = MySampler(backend="cuda",model = adj_model)
obs = sampler.observables(ordmat)
print("observed observables: ", obs)
ordmat = ordmat.to(device)
parlist = sampler.param_run(graph=ordmat,
                      observables=obs,
                      params=betas,
                      niter=100000,
                      params_update_every=3,
                      save_every=50,
                      save_params=True,
                      alpha=0.001,                      
                      min_change = 0.01)

params = [p.cpu().numpy() for p in parlist[0]]
parlist_np = np.array(params)
print(parlist_np[-1])
print(parlist_np.shape)

w = 10
h = 8
scale = 0.6

w = int(w * scale)
h = int(h * scale)

plt.figure(figsize = (parlist_np.shape[1] * w, h))
for p in range(parlist_np.shape[1]):
    plt.subplot(1,parlist_np.shape[1], p + 1)
    plt.plot(parlist_np[:,p], '.-')

plt.savefig('parameter_evolution.png')


params_for_estimates = torch.stack(parlist[0][:-100:-1]).mean(axis = 0)
# run in sample mode, without parameter modifications
graphs, observables = sampler.sample_run(graph=ordmat,
                      observables=obs,
                      params=params_for_estimates,
                      niter=100000,
                      save_every=10,
                      burn_in=0.5)

print('original observables: ', obs)