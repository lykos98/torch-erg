from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.samplers import GWGSampler, MHSampler
from src.torch_erg.utils import laplacian_matrix
import torch
import numpy as np

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
# parsing the .m file into a suitable format, and perform some sanity checks
import matplotlib.pyplot as plt

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

class MySampler(GWGSampler):
    def __init__(self, backend: str):
        super().__init__(backend)
    def observables(self, mtx):
        edges = torch.sum(mtx)/2
        triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
        ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
        return(torch.stack([edges, triangles,ac]))
        
betas = torch.tensor([0., 0., 0.], dtype=float)
sampler = MySampler(backend="cuda")
obs = sampler.observables(ordmat)
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



