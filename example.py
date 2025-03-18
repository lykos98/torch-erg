from torch_erg import load_pglib_opf as lp
from torch_erg.samplers import GWGSampler
import torch
import numpy as np

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
# parsing the .m file into a suitable format, and perform some sanity checks
import matplotlib.pyplot as plt

# test grid of reference, can either be 30_ieee, 118_ieee or 300_ieee
name = '300_ieee'

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

betas = torch.tensor([0, 0, 0], dtype=float)
sampler = GWGSampler(backend="cuda")
obs = sampler.observables(ordmat)
parlist = sampler.run(graph=ordmat,
                      observables=obs,
                      params=betas,
                      niter=5000,
                      params_update_every=3,
                      save_every=10,
                      save_params=True,
                      alpha=0.01,                      
                      min_change = 0.01)

params = [p.cpu().numpy() for p in parlist[0]]
parlist_np = np.array(params)
print(parlist_np[-1])
print(parlist_np.shape)
for p in range(parlist_np.shape[1]):
    plt.figure()
    plt.plot(parlist_np[:,p], '.-')
    plt.savefig(f"convergence_plot_{p}.png")



