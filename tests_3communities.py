import torch
import pickle
import graph_generation
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import pickle
import numpy as np
import torch
from tqdm import tqdm

from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.samplers import ChimeraSamplerFeatures, MHSamplerFeatures, GraphTuple

def basic_observables(graph_tuple: GraphTuple) -> torch.Tensor:
    mtx = graph_tuple.adj
    feat = graph_tuple.node_features

    nn = torch.sum(feat, axis=0)
    mm = (feat.T) @ mtx @ feat
    # val = 10 * mm[0,0]/torch.sum(feat, axis = 0)[0]
    indices = torch.triu_indices(mm.shape[0], mm.shape[1], offset=1)

    vals = mm[indices[0], indices[1]]
    mm = mm / nn

    in_community = torch.prod(torch.diag(mm))
    out_community = -torch.prod(vals)
    edges = torch.sum(mtx) / 2
    triangles = torch.trace(mtx @ mtx @ mtx) / 6

    list_obs = []
    for number in nn:
        list_obs.append(number)
    list_obs.append(in_community)
    list_obs.append(out_community)
    list_obs.append(edges)
    list_obs.append(triangles)
    return torch.stack(list_obs)

class MySampler(ChimeraSamplerFeatures):
    def __init__(self, p_edge=0.8, backend="cpu"):
        super().__init__(p_edge=p_edge, backend=backend)

    def observables(self, graph_tuple):
        return basic_observables(graph_tuple)

graphs, metadata = graph_generation.community_sbm.load_community_dataset("data/community_sbm/params_5_30_0.8_0.02_4/graphs.pkl")
# Compute test observables
test_obs = basic_observables(graphs[1])
print(test_obs)

# Instantiate sampler
sampler = MySampler()

graph = graphs[1]
obs = sampler.observables(graph)
params = torch.zeros_like(test_obs)
niter = 500000
params_update_every = 5
save_every = 250
alpha = 0.0002
min_change = 0.001
tot_accept = 1000000

params_hist, graph_hist = sampler.param_run(
    graph=graph,
    observables=obs,
    params=params,
    niter=niter,
    params_update_every=params_update_every,
    save_every=save_every,
    save_params=True,
    alpha=alpha,
    min_change=min_change,
    tot_accept=tot_accept,
    verbose_level=0
)

params_final = params_hist[-1]
print(params_final)

return_obs, return_graph = sampler.sample_run(
    graph=graph,
    params=params_final,
    niter=100000,
    save_every=save_every,
    burn_in=0.3
)

# Visualize final graph
graph_generation.visualization.show_community_graph(return_graph[-1])
plt.savefig('final_graph.png')
plt.close()
