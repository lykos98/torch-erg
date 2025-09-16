from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.samplers import GWGSampler, MHSampler, MHSampler_Hard
import torch
import numpy as np
import os
import pickle
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# parsing the .m file into a suitable format, and perform some sanity checks

import matplotlib.pyplot as plt

seed = np.random.randint(500)

# test grid of reference, can either be 30_ieee, 118_ieee or 300_ieee

name = '30_ieee'



mc_type = 'MH_'

if mc_type == 'GWG_':
    class MySampler(GWGSampler):
        def __init__(self, backend: str):
            super().__init__(backend)
        def observables(self, mtx):
            edges = torch.sum(mtx)/2
            triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
            #ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
            return(torch.stack([edges, triangles]))
elif mc_type == 'MH_':
    class MySampler(MHSampler_Hard):
        def __init__(self, backend: str):
            super().__init__(backend)
        def observables(self, mtx):
            edges = torch.sum(mtx)/2
            triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
            #ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
            return(torch.stack([edges, triangles]))

else:
    print('Invalid sampler')

sampler = MySampler(backend="cuda")



ordmat, ordlist, buslist, countlist = lp.pow_parser(name)

ordmat = torch.tensor(ordmat)
ordlist = torch.tensor(ordlist)
buslist = torch.tensor(buslist)
countlist = torch.tensor(countlist)

G_sparse = ordmat.cpu().numpy()  # Create sparse matrix
n_components = connected_components(csr_matrix(G_sparse))
print("Number of connected components in the graph:", n_components[0])
max_iter = 500000
betas = torch.tensor([0,  0], dtype=float)

obs = sampler.observables(ordmat)


parlist = sampler.param_run(graph=ordmat,
                      observables=obs,
                      params=betas,
                      niter=1000000,
                      params_update_every=3,
                      save_every=50,
                      save_params=True,
                      alpha=0.001,                      
                      min_change = 0.005)

params_for_estimates = torch.stack(parlist[0][:-100:-1]).mean(axis = 0)

print(params_for_estimates)

graphs_list, obs_list = sampler.sample_run(graph=ordmat,
                      observables=obs,
                      params=params_for_estimates,
                      niter=max_iter,
                      save_every=50,
                      burn_in = 0.3
                      )



ensemble_results = {'graphs' : graphs_list, 'obs' : obs_list, 'real_obs': obs, 'iterations': max_iter}

results_path = os.path.join('results', mc_type +name+'_ensemble_'+ str(seed)+'_.pkl')

with open(results_path, "wb") as f:
    pickle.dump(ensemble_results, f)


print("number of generated graphs: ", len(graphs_list))