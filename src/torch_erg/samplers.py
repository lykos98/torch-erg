import torch
import numpy as np

from .utils import *
from tqdm import tqdm
from typing import Union, Tuple

from abc import ABC, abstractmethod

class BaseSampler(ABC):
    def __init__(self, backend = "cuda"):
        self.params = 0
        self.states = []
        self.backend = backend
        if backend == "cuda" and not torch.cuda.is_available():
            print("CUDA backend not available falling back to cpu")
            self.backend = "cpu"
        if not (backend in ["cpu", "cuda"]):
            print("Backend not recognized falling back to cpu")
            self.backend = "cpu"

    
    def _hamiltonian(self, observables, params) -> float:
        return torch.dot(observables, params)

    @abstractmethod
    def observables(self, graph: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def proposal(self,  graph: torch.tensor, 
                        params: torch.tensor, 
                        obs: torch.tensor,
                        hamiltonian: torch.tensor) -> Tuple[torch.tensor, float]:
        pass

    def _update_parameters(self,observed, reference, parameters, alpha, min_change):
        updated_params = parameters.clone()  # Clone to avoid modifying torche original
        updated_params.detach_()

        min_change_tensor = torch.ones_like(updated_params, dtype = torch.float32, device = self.backend) * min_change
        change = - alpha * torch.max(updated_params.abs(), min_change_tensor) * torch.sign(observed - reference)
        updated_params += change
        return updated_params
    
    def run(self,   graph:               torch.tensor, 
                    observables:         torch.tensor, 
                    params:              torch.tensor,
                    niter:               int, 
                    params_update_every: int,
                    save_every:          int, 
                    save_params:         bool,
                    alpha:               float,
                    min_change:          float,
                    save_graph:          bool = True,
                    verbose_level:       int = 0
            ) -> list[torch.tensor]:

        start_graph  = graph.clone().to(self.backend)
        start_params = params.clone().to(self.backend)
        start_obs    = observables.clone().to(self.backend)


        current_graph  = start_graph.clone().detach().to(self.backend)
        current_params = start_params.clone().detach().to(self.backend)
        current_obs    = start_obs.clone().detach().to(self.backend)

        #bootstrap
        current_graph.requires_grad = True
        current_obs         = self.observables(current_graph)
        current_hamiltonian = self._hamiltonian(current_obs, current_params)

        accepted_steps   = 0

        return_params = []
        return_graph  = []
        
        #some logging info, for example using gpu or cpu
        #graph dimensions, paramters
        update_steps = 0
        rejected_samples = 0

        for it in tqdm(range(niter)):
            #uniform indexes selection
            new_graph, acceptance_prob = self.proposal(current_graph, current_params, current_obs, current_hamiltonian)


            p = torch.rand(1).item()

            if (p < acceptance_prob):

                new_graph.requires_grad_()

                current_graph = new_graph
                current_obs = self.observables(new_graph)
                current_hamiltonian = self._hamiltonian(current_obs, current_params)

                accepted_steps += 1
                if accepted_steps % params_update_every == 0:
                    update_steps += 1
                    current_params = self._update_parameters(current_obs, start_obs, current_params, alpha, min_change)
            else:
                rejected_samples += 1
            if accepted_steps % save_every == 0:
                if save_graph:
                    return_graph.append(current_graph.clone().detach())
                if save_params:
                    return_params.append(current_params.clone().detach())

        #some logging info also here fraction of samples accepted
        
        return_params.append(current_params.clone().detach())
        return_graph.append(current_graph.clone().detach())
        return return_params, return_graph



class GWGSampler(BaseSampler):
    def __init__(self, backend: str):
        super().__init__(backend)

    def _d(self,x):
        with torch.no_grad():
            d = -(2*x - 1) * x.grad
        return d
    def proposal(self,mtx, obs, params, ham):
        if mtx.grad is None: 
            ham.backward(retain_graph = True)

        dx = self._d(mtx)
        with torch.no_grad():
            # TODO: 
            # ask this to fjack
            q_ix = torch.nn.functional.softmax(dx.ravel(), dim=0)
            # Vectorized sampling of a single edge
            sampled_index = torch.multinomial(q_ix, 1).item()
            i = sampled_index // mtx.shape[1]
            j = sampled_index % mtx.shape[1]

            new_mtx = mtx.clone().detach()
            new_mtx[i,j] = 1 - new_mtx[i,j]
            new_mtx[j,i] = new_mtx[i,j]

        new_mtx.requires_grad_()

        new_obs = self.observables(new_mtx)
        new_ham = self._hamiltonian(new_obs,params)

        new_ham.backward(retain_graph = True)

        dx = self._d(new_mtx)
        with torch.no_grad():
            q_ix_prime = torch.nn.functional.softmax(dx.ravel(), dim=0)

        qq = torch.exp(new_ham - ham) * q_ix_prime[i * mtx.shape[1] + j]/q_ix[i * mtx.shape[1] + j]
        acceptance_prob = min(1, qq.item())   
        
        return new_mtx, acceptance_prob

    def observables(self, mtx):
        edges = torch.sum(mtx)/2
        triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
        ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
        return(torch.stack([edges, triangles, ac]))




