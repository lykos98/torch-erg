import torch
import numpy as np

from .utils import *
from tqdm import tqdm
from typing import Union

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
                        hamiltonian: torch.tensor) -> torch.tensor:
        pass

    def _update_parameters(self,observed, reference, parameters, alpha, min_change):
        updated_params = parameters.clone()  # Clone to avoid modifying torche original
        updated_params.detach_()

        min_change_tensor = torch.ones_like(updated_params, dtype = torch.float32, device = self.backend) * min_change
        change = alpha * torch.max(updated_params.abs(), min_change_tensor) * - torch.sign(observed - reference)
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
        n = graph.shape[0]  # Number of nodes

        start_graph  = graph.clone().to(self.backend)
        start_params = params.clone().to(self.backend)
        start_obs    = observables.clone().to(self.backend)


        current_graph  = start_graph.clone().to(self.backend)
        current_params = start_params.clone().to(self.backend)
        current_obs    = start_obs.clone().to(self.backend)

        #bootstrap
        current_graph.requires_grad = True
        current_obs         = self.observables(current_graph)
        current_hamiltonian = self._hamiltonian(current_obs, current_params)

        accepted_steps   = 0

        return_params = []
        return_graph  = []
        
        old_params = current_params
        #some logging info, for example using gpu or cpu
        #graph dimensions, paramters
        update_steps = 0
        rejected_samples = 0

        for it in tqdm(range(niter)):
            #uniform indexes selection
            new_graph = self.proposal(current_graph, current_params, current_obs, current_hamiltonian)

            new_graph.requires_grad = True
            new_obs = self.observables(new_graph)
            new_hamiltonian = self._hamiltonian(new_obs, current_params)

            p = torch.rand(1).item()
            exp_p = torch.exp(new_hamiltonian - current_hamiltonian)

            if (p < exp_p):
                current_graph = new_graph
                current_hamiltonian = new_hamiltonian
                current_obs = new_obs

                accepted_steps += 1
                if accepted_steps % params_update_every == 0:
                    update_steps += 1
                    old_params = current_params
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

    def proposal(self,mtx, obs, params, hamiltonian):
        if mtx.grad is None: 
            hamiltonian.backward(retain_graph = True)
            #hamiltonian.backward()
        return gwg_move(mtx)
    def observables(self, mtx):
        edges = torch.sum(mtx)/2
        triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
        ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
        return(torch.stack([edges, triangles, ac]))




