from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.samplers import GWGSampler, MHSampler
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

betas = torch.tensor([-3.01565308,  0.7548087,   6.63389789], dtype=float)
sampler = GWGSampler(backend="cuda")
obs = sampler.observables(ordmat)
graphs_list, obs_list = sampler.sample_run(graph=ordmat,
                      observables=obs,
                      params=betas,
                      niter=300000,
                      save_every=10,
                      burn_in = 5000
                      )


ensemble_results = {'graphs' : graphs_list, 'obs' : obs_list}

results_path = os.path.join('results', name+'_ensemble_'+ str(seed)+'_.pkl')

with open(results_path, "wb") as f:
    pickle.dump(ensemble_results, f)