import torch
import numpy as np

from .utils import *
from tqdm import tqdm
from typing import Union, Tuple, Optional, Any
import torch.nn as nn
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from dataclasses import dataclass

from abc import ABC, abstractmethod

@dataclass
class GraphTuple:
    adj: torch.Tensor
    node_features: Optional[torch.Tensor] = None
    edge_features: Optional[torch.Tensor] = None

    def clone(self):
        return GraphTuple(
            adj=self.adj.clone(),
            node_features=self.node_features.clone() if self.node_features is not None else None,
            edge_features=self.edge_features.clone() if self.edge_features is not None else None
        )

    def detach(self):
        return GraphTuple(
            adj=self.adj.detach(),
            node_features=self.node_features.detach() if self.node_features is not None else None,
            edge_features=self.edge_features.detach() if self.edge_features is not None else None
        )

    def to(self, *args, **kwargs):
        return GraphTuple(
            adj=self.adj.to(*args, **kwargs),
            node_features=self.node_features.to(*args, **kwargs) if self.node_features is not None else None,
            edge_features=self.edge_features.to(*args, **kwargs) if self.edge_features is not None else None
        )

    @property
    def requires_grad(self):
        return self.adj.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.adj.requires_grad = value
        if self.node_features is not None:
            self.node_features.requires_grad = value
        if self.edge_features is not None:
            self.edge_features.requires_grad = value

    def requires_grad_(self, requires_grad=True):
        self.adj.requires_grad_(requires_grad)
        if self.node_features is not None:
            self.node_features.requires_grad_(requires_grad)
        if self.edge_features is not None:
            self.edge_features.requires_grad_(requires_grad)
        return self

    # Helper methods for interoperability with tensor-based logic
    def size(self, *args, **kwargs):
        return self.adj.size(*args, **kwargs)

    @property
    def shape(self):
        return self.adj.shape

    @property
    def device(self):
        return self.adj.device

    @property
    def dtype(self):
        return self.adj.dtype

    def __getitem__(self, key):
        return self.adj[key]

    def __setitem__(self, key, value):
        self.adj[key] = value

    @property
    def grad(self):
        return self.adj.grad

    def __mul__(self, other):
        return self.adj * other

    def __rmul__(self, other):
        return other * self.adj

    def __add__(self, other):
        return self.adj + other

    def __radd__(self, other):
        return other + self.adj

    def __sub__(self, other):
        return self.adj - other

    def __rsub__(self, other):
        return other - self.adj

class BaseSampler(ABC):
    def __init__(self, backend = "cuda", model=None):
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
    def observables(self, graph: Union[torch.Tensor, GraphTuple]) -> torch.Tensor:
        pass

    @abstractmethod
    def proposal(self,  graph: Union[torch.Tensor, GraphTuple], 
                        obs: torch.Tensor,
                        params: torch.Tensor, 
                        hamiltonian: torch.Tensor,
                        mod_ratio: bool = False) -> Tuple[Union[torch.Tensor, GraphTuple], float]:
        pass

    def _update_parameters(self,observed, reference, parameters, alpha, min_change):
        updated_params = parameters.clone()  # Clone to avoid modifying torche original
        updated_params.detach_()

        min_change_tensor = torch.ones_like(updated_params, dtype = torch.float32, device = self.backend) * min_change
        change = - alpha * torch.max(updated_params.abs(), min_change_tensor) * torch.sign(observed - reference)
        updated_params += change
        return updated_params
    
    def param_run(self, graph:               Union[torch.Tensor, GraphTuple], 
                        observables:         torch.Tensor, 
                        params:              torch.Tensor,
                        niter:               int, 
                        params_update_every: int,
                        save_every:          int, 
                        save_params:         bool,
                        alpha:               float,
                        min_change:          float,
                        save_graph:          bool = True,
                        tot_accept:          int = 10000000000,
                        verbose_level:       int = 0
            ) -> list[torch.Tensor]:

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

        self.accepted_steps   = 0
        self.rejected_samples = 0

        return_params = []
        return_graph  = []
        
        #some logging info, for example using gpu or cpu
        #graph dimensions, paramters
        update_steps = 0

        for it in tqdm(range(niter)):
            #uniform indexes selection
            new_graph, acceptance_prob = self.proposal(current_graph, current_obs, current_params, current_hamiltonian)


            p = torch.rand(1).item()

            if (p < acceptance_prob):

                new_graph.requires_grad_()

                current_graph = new_graph
                current_obs = self.observables(new_graph)
                current_hamiltonian = self._hamiltonian(current_obs, current_params)
                #print("current obs are:", current_obs)

                self.accepted_steps += 1
                if self.accepted_steps % params_update_every == 0:
                    update_steps += 1
                    current_params = self._update_parameters(current_obs, start_obs, current_params, alpha, min_change)
                 #   print("params now are: ",current_params)
            else:
                self.rejected_samples += 1
            if self.accepted_steps % save_every == 0:
                #print('number of effective updates is: ', update_steps)
                if save_graph:
                    return_graph.append(current_graph.clone().detach())
                if save_params:
                    return_params.append(current_params.clone().detach())
            if self.accepted_steps > tot_accept:
                break

        #some logging info also here fraction of samples accepted
        print('number of accepted steps is: ', self.accepted_steps)
        print('number of rejected samples: ', self.rejected_samples)
        print('number of effective updates is: ', update_steps)
        
        return_params.append(current_params.clone().detach())
        return_graph.append(current_graph.clone().detach())
        return return_params, return_graph
    
    def sample_run(self, graph:               Union[torch.Tensor, GraphTuple], 
                         params:              torch.Tensor,
                         niter:               int, 
                         save_every:          int,
                         burn_in:             float = 0.,
                         verbose_level:       int = 0
            ) -> list[torch.Tensor]:

        assert burn_in >= 0. and burn_in < 1., "Invalid burn in fraction, should be [0., 1.)"

        burn_in_iter = int(burn_in * niter)
        start_graph  = graph.clone().to(self.backend)
        start_params = params.clone().to(self.backend)

        current_graph  = start_graph.clone().detach().to(self.backend)
        current_params = start_params.clone().detach().to(self.backend)
        current_obs    = self.observables(start_graph)

        #bootstrap
        current_graph.requires_grad = True
        current_obs         = self.observables(current_graph)
        current_hamiltonian = self._hamiltonian(current_obs, current_params)

        accepted_steps   = 0

        return_graph  = []
        return_obs = []
        #some logging info, for example using gpu or cpu
        #graph dimensions, paramters
        
        rejected_samples = 0

        for it in tqdm(range(niter)):
            #uniform indexes selection
            new_graph, acceptance_prob = self.proposal(current_graph, current_obs, current_params, current_hamiltonian)


            p = torch.rand(1).item()

            if (p < acceptance_prob):

                new_graph.requires_grad_()

                current_graph = new_graph
                current_obs = self.observables(new_graph)
                current_hamiltonian = self._hamiltonian(current_obs, current_params)
                #print("current obs are:", current_obs)

                accepted_steps += 1
            else:
                rejected_samples += 1
            if it > burn_in_iter:
                if accepted_steps % save_every == 0:
                    return_graph.append(current_graph.clone().detach())
                    return_obs.append(self.observables(current_graph).clone().detach())

        #some logging info also here fraction of samples accepted
        
        print('number of accepted steps is: ', accepted_steps)
        print('number of rejected samples: ', rejected_samples)
        return_graph.append(current_graph.clone().detach())
        return_obs.append(self.observables(current_graph).clone().detach())
        tt = torch.stack(return_obs).mean(axis = 0)
        print("Mean obs: ", tt)
        return return_obs, return_graph
    

class MHSampler(BaseSampler, ABC):
    def __init__(self, backend: str):
        super().__init__(backend)
    
    def proposal(self,mtx, obs, params, ham):
        
        new_mtx, i, j = unif_move(mtx)
        new_obs = self.observables(new_mtx)
        #print("obs are: ", new_obs)
        #print("params are: ", params)
        new_ham = self._hamiltonian(new_obs,params)

        


        qq = torch.exp(new_ham - ham) 
        #print("new ham is: ", new_ham)
        #print("old ham is: ", ham)
        #print("qq is: ", qq)
        acceptance_prob = min(1, qq.item())   
        
        return new_mtx, acceptance_prob

    
    def observables(self, mtx):
        pass

class MHSamplerFeatures(BaseSampler, ABC):
    """ 
        For edge feature suppose I have the features one hot encoded, so 
        I know what is the distribution at each time?
    """
    def __init__(self, backend: str, p_edge: float = 0.8):
        super().__init__(backend)
        self.p_edge = p_edge
    

    def __edge_unif_move(self, graph_tuple: GraphTuple):
        n = graph_tuple.adj.size(0)
        i, j = rand_indexes(n)
        new_graph_tuple = graph_tuple.clone().detach()
        new_graph_tuple.adj[i,j] = 1 - new_graph_tuple.adj[i,j]
        new_graph_tuple.adj[j,i] = new_graph_tuple.adj[i,j]
        return new_graph_tuple

    def __node_feature_unif_move(self, graph_tuple: GraphTuple):
        n = graph_tuple.node_features.size(0)
        k = graph_tuple.node_features.size(1)
        node_idx = np.random.choice(n) 
        # choose a different one
        one_idx  = torch.argmax(graph_tuple.node_features[node_idx]).item()
        feature_idx = (1 + np.random.choice(k - 1) + one_idx) % k 

        new_graph_tuple = graph_tuple.clone().detach()
        new_graph_tuple.node_features[node_idx] = 0
        new_graph_tuple.node_features[node_idx,feature_idx] = 1

        return new_graph_tuple
        

    def proposal(self, graph_tuple: GraphTuple, 
                       obs: torch.Tensor, 
                       params: torch.Tensor,
                       ham: torch.Tensor):

        # choose if I have to go one step into the adj or into the 
        # features

        if np.random.rand() < self.p_edge:
            # edge path
            new_graph_tuple = self.__edge_unif_move(graph_tuple)
        else:
            # feature path
            new_graph_tuple = self.__node_feature_unif_move(graph_tuple)
        
        new_obs = self.observables(new_graph_tuple)
        new_ham = self._hamiltonian(new_obs,params)
        qq = torch.exp(new_ham - ham) 
        #print("new ham is: ", new_ham)
        #print("old ham is: ", ham)
        #print("qq is: ", qq)
        acceptance_prob = min(1, qq.item())   
        
        return new_graph_tuple, acceptance_prob
    
    def observables(self, mtx):
        pass



class MHSampler_Hard(BaseSampler, ABC):
    def __init__(self, backend: str):
        super().__init__(backend)
    
    def proposal(self,mtx, obs, params, ham):
        
        new_mtx, i, j = unif_move(mtx)
        G_sparse = new_mtx.cpu().numpy()  # Create sparse matrix
        n_components = connected_components(csr_matrix(G_sparse))[0]

        if n_components>1:
            acceptance_prob = 0
            return new_mtx, acceptance_prob

        else:
            
            new_obs = self.observables(new_mtx)
            #print("obs are: ", new_obs)
            #print("params are: ", params)
            new_ham = self._hamiltonian(new_obs,params)

            


            qq = torch.exp(new_ham - ham) 
            #print("new ham is: ", new_ham)
            #print("old ham is: ", ham)
            #print("qq is: ", qq)
            acceptance_prob = min(1, qq.item())   
            
            return new_mtx, acceptance_prob

    
    def observables(self, mtx):
        pass




class GWGSampler(BaseSampler, ABC):
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
        torch.diagonal(dx, 0).zero_()
        with torch.no_grad():
            # TODO: 
            # ask this to fjack
            q_ix = torch.nn.functional.softmax(dx.ravel(), dim=0)
            # Vectorized sampling of a single edge
            #sampled_index = torch.multinomial(q_ix, 1).item()
            i, j = index_ravel_sampler(q_ix, mtx)
            new_mtx = mtx.clone().detach()
            new_mtx[i,j] = 1 - new_mtx[i,j]
            new_mtx[j,i] = new_mtx[i,j]

        new_mtx.requires_grad_()

        new_obs = self.observables(new_mtx)

        #print("obs are: ", new_obs)
        #print("params are: ", params)
        new_ham = self._hamiltonian(new_obs,params)

        new_ham.backward(retain_graph = True)

        dx = self._d(new_mtx)
        torch.diagonal(dx, 0).zero_()
        with torch.no_grad():
            q_ix_prime = torch.nn.functional.softmax(dx.ravel(), dim=0)

        qq = torch.exp(new_ham - ham) * q_ix_prime[i * mtx.shape[1] + j]/q_ix[i * mtx.shape[1] + j]
        #print("new ham is: ", new_ham)
        #print("old ham is: ", ham)
        #print("qq is: ", qq)
        acceptance_prob = min(1, qq.item())   
        
        return new_mtx, acceptance_prob

    
    def observables(self, mtx):
        pass




class GWG_Hybrid_Sampler(BaseSampler, ABC):
    def __init__(self, backend: str, model: nn.Module, mod_ratio: bool = False):
        super().__init__(backend, model)
        self.model = model
        self.mod_ratio = mod_ratio

        self.model.to(self.backend)

    def _d(self,x):
        with torch.no_grad():
            g = x.grad
            d = -(2*x - 1) * g
            #print('Gradient norm:', torch.norm(g))
        return d
    def proposal(self,mtx, obs, params, ham):
        
        mtx= mtx.to(self.backend)
        ham = ham.to(self.backend)
        mod = self.model(mtx)
        comp_ham = ham + mod

        #print('Model ratio: ', mod / comp_ham)

        if mtx.grad is None: 
            comp_ham.backward(retain_graph = True)

        dx = self._d(mtx)
        torch.diagonal(dx, 0).zero_()
        with torch.no_grad():
            # TODO: 
            # ask this to fjack
            q_ix = torch.nn.functional.softmax(dx.ravel(), dim=0)
            # Vectorized sampling of a single edge
            #sampled_index = torch.multinomial(q_ix, 1).item()
            i, j = index_ravel_sampler(q_ix, mtx)
            new_mtx = mtx.clone().detach()
            new_mtx[i,j] = 1 - new_mtx[i,j]
            new_mtx[j,i] = new_mtx[i,j]

        new_mtx.requires_grad_()

        new_obs = self.observables(new_mtx)

        #print("obs are: ", new_obs)
        #print("params are: ", params)
        new_ham_base = self._hamiltonian(new_obs,params)

        new_ham_model = self.model(new_mtx)

        new_ham = new_ham_base + new_ham_model

        mod_ratio = torch.abs(new_ham_model) / (torch.abs(new_ham_model)+torch.abs(new_ham_base)+1e-30)

        #print('Model ratio contribution: ', mod_ratio)
        new_ham.backward(retain_graph = True)

        dx = self._d(new_mtx)
        torch.diagonal(dx, 0).zero_()
        with torch.no_grad():
            q_ix_prime = torch.nn.functional.softmax(dx.ravel(), dim=0)

        qq = torch.exp(new_ham - comp_ham) * q_ix_prime[i * mtx.shape[1] + j]/q_ix[i * mtx.shape[1] + j]
        #print("new ham is: ", new_ham)
        #print("old ham is: ", comp_ham)
        #print("qq is: ", qq)
        acceptance_prob = min(1, qq.item())
        if self.mod_ratio:
            acceptance_prob = acceptance_prob * (mod_ratio.item()) 
            return new_mtx, acceptance_prob  
        #print('Acceptance prob:', acceptance_prob)
        return new_mtx, acceptance_prob

    
    def observables(self, mtx):
        pass



######################################
##### COSE DA TESTARE VIBECODATE #####
######################################

def g_balanced(r: torch.Tensor, kind: str = "sqrt"):
    # Locally balanced weights g(r) with g(t)=t g(1/t)
    if kind == "sqrt":
        return torch.sqrt(r.clamp_min(1e-30))
    elif kind in ("a_over_1p_a", "ratio"):
        return r / (1.0 + r)
    else:
        raise ValueError(f"Unknown g kind: {kind}")

def erg_energy(observables: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    # Scalar energy E = - <obs, theta> if π ∝ exp(-E). Your code used dot(obs, params) as "hamiltonian".
    # Keep sign consistent with your MH ratio exp(new_ham - old_ham). We'll treat 'ham' as energy.
    return torch.dot(observables, params)

def target_energy_base(mtx, params, observables_fn):
    obs = observables_fn(mtx)
    return erg_energy(obs, params)

def target_energy_hybrid(mtx, params, observables_fn, model: nn.Module):
    base = target_energy_base(mtx, params, observables_fn)
    model_term = model(mtx)
    if model_term.dim() > 0:
        model_term = model_term.view(()).to(base)
    return base + model_term

def iter_upper_tri_indices(n):
    # yields (i,j) for i<j
    for i in range(n):
        for j in range(i+1, n):
            yield i, j




class DLMC_Sampler(BaseSampler):
    """
    Discrete Langevin Monte Carlo (DLMC) for binary sites (edges), factorized across sites.
    - Uses exact 2x2 transition (Eq. 37).
    - Proposal: sample all edges in parallel from row xt_n of P^h_n; then MH correct with product ratio.
    - Energy = ERG only (attach deep model with DLMC_Hybrid_Sampler).
    """
    def __init__(self, backend: str, h: float = 1.0, g_kind: str = "sqrt",
                 w: float = 1.0, changed_only: bool = False, local_delta_fn=None):
        super().__init__(backend)
        self.h = float(h)
        self.g_kind = g_kind
        self.w = float(w)
        self.changed_only = changed_only
        self.local_delta_fn = local_delta_fn  # optional: (mtx,i,j,new_bit)-> energy for that local flip

    def _Q_rates_binary(self, E0, E1):
        """
        Given energies when site=0 (E0) and site=1 (E1), build alpha_,beta_:
        alpha_ = Q(1->2) = w * g(π(1)/π(0)) = w * g(exp(-(E1-E0)))
        beta_ = Q(2->1) = w * g(π(0)/π(1)) = w * g(exp(-(E0-E1)))
        """
        dE = (E1 - E0)
        r10 = torch.exp(-(dE))     # π(1)/π(0)
        r01 = torch.exp(-(-dE))    # π(0)/π(1)
        g1 = g_balanced(r10, self.g_kind)
        g0 = g_balanced(r01, self.g_kind)
        alpha_ = self.w * g1
        beta_ = self.w * g0
        return alpha_, beta_

    @torch.no_grad()
    def _build_proposal_and_sample(self, mtx, params, observables_fn, base_energy_fn):
        """
        Build factorized proposal rows for every edge and sample a full candidate matrix y.
        Returns:
          new_mtx, log_q_fwd (sum over sites), cache with per-site info to compute reverse probs.
        """
        n = mtx.shape[0]
        device = mtx.device
        i_upper, j_upper = torch.triu_indices(n, n, offset=1, device=device)
        num_edges = len(i_upper)

        cur_bits = mtx[i_upper, j_upper]

        # Create a batch of matrices for energy calculation
        batch_mtx = mtx.unsqueeze(0).expand(2 * num_edges, -1, -1).clone()

        edge_indices_i = i_upper.repeat(2)
        edge_indices_j = j_upper.repeat(2)
        
        batch_indices = torch.arange(2 * num_edges, device=device)

        bits_to_set = torch.zeros(2 * num_edges, device=device)
        bits_to_set[num_edges:] = 1.0

        batch_mtx[batch_indices, edge_indices_i, edge_indices_j] = bits_to_set
        batch_mtx[batch_indices, edge_indices_j, edge_indices_i] = bits_to_set
        
        # Compute energies using vmap for batching.
        batch_obs = torch.vmap(observables_fn)(batch_mtx)
        energies = batch_obs @ params

        E_all = energies.view(2, num_edges)
        E0s, E1s = E_all[0], E_all[1]

        # Compute Q rates
        alphas, betas = self._Q_rates_binary(E0s, E1s)

        # Compute transition probabilities
        s = (alphas + betas).clamp_min(1e-30)
        e = torch.exp(-s * self.h)
        
        p00 = betas/s + (alphas/s) * e
        p01 = alphas/s - (alphas/s) * e
        p10 = betas/s - (betas/s) * e
        p11 = alphas/s + (betas/s) * e
        
        p_trans = torch.stack([p00, p01, p10, p11], dim=1).view(num_edges, 2, 2)

        # Select the correct row of transition probabilities
        rows = p_trans[torch.arange(num_edges), cur_bits.long(), :]
        rows = rows.clamp(min=1e-30)
        rows = rows / rows.sum(dim=1, keepdim=True)

        # Sample new bits
        new_bits = torch.multinomial(rows, 1).squeeze(-1)

        # Compute forward log probability
        log_q_fwd = torch.log(rows[torch.arange(num_edges), new_bits]).sum().item()

        # Construct new matrix
        new_mtx = mtx.clone()
        new_mtx[i_upper, j_upper] = new_bits.float()
        new_mtx[j_upper, i_upper] = new_bits.float()

        # Cache info for reverse probability calculation
        site_cache = {
            'i_upper': i_upper, 'j_upper': j_upper,
            'cur_bits': cur_bits, 'new_bits': new_bits,
        }

        return new_mtx, log_q_fwd, site_cache

    @torch.no_grad()
    def _log_q_reverse(self, new_mtx, old_mtx, params, observables_fn, site_cache):
        """
        Compute sum log P^h_n(y_n -> x_n) under state y (reverse rows depend on y context).
        """
        n = new_mtx.shape[0]
        device = new_mtx.device
        
        i_upper = site_cache['i_upper']
        j_upper = site_cache['j_upper']
        cur_bits = site_cache['cur_bits']
        new_bits = site_cache['new_bits']
        num_edges = len(i_upper)

        # Create batch of matrices based on new_mtx for reverse probabilities
        batch_mtx = new_mtx.unsqueeze(0).expand(2 * num_edges, -1, -1).clone()

        edge_indices_i = i_upper.repeat(2)
        edge_indices_j = j_upper.repeat(2)
        
        batch_indices = torch.arange(2 * num_edges, device=device)

        bits_to_set = torch.zeros(2 * num_edges, device=device)
        bits_to_set[num_edges:] = 1.0

        batch_mtx[batch_indices, edge_indices_i, edge_indices_j] = bits_to_set
        batch_mtx[batch_indices, edge_indices_j, edge_indices_i] = bits_to_set
        
        # Compute energies for the batch using vmap.
        batch_obs = torch.vmap(observables_fn)(batch_mtx)
        energies = batch_obs @ params

        E_all_y = energies.view(2, num_edges)
        E0s_y, E1s_y = E_all_y[0], E_all_y[1]

        # Compute Q rates for reverse
        alphas_y, betas_y = self._Q_rates_binary(E0s_y, E1s_y)

        # Compute transition probabilities for reverse
        s_y = (alphas_y + betas_y).clamp_min(1e-30)
        e_y = torch.exp(-s_y * self.h)
        
        p00_y = betas_y/s_y + (alphas_y/s_y) * e_y
        p01_y = alphas_y/s_y - (alphas_y/s_y) * e_y
        p10_y = betas_y/s_y - (betas_y/s_y) * e_y
        p11_y = alphas_y/s_y + (betas_y/s_y) * e_y
        
        p_trans_y = torch.stack([p00_y, p01_y, p10_y, p11_y], dim=1).view(num_edges, 2, 2)

        # Select rows based on new_bits (starting state for reverse)
        rows_y = p_trans_y[torch.arange(num_edges), new_bits.long(), :]
        rows_y = rows_y.clamp(min=1e-30)
        rows_y = rows_y / rows_y.sum(dim=1, keepdim=True)

        # Probability of transitioning from new_bit back to cur_bit
        log_q_rev = torch.log(rows_y[torch.arange(num_edges), cur_bits.long()]).sum().item()
        
        return log_q_rev

    def proposal(self, mtx, obs, params, ham):
        # Build candidate with factorized DLMC
        mtx = mtx.to(self.backend)
        params = params.to(self.backend)
        new_mtx, log_q_fwd, cache = self._build_proposal_and_sample(mtx, params, self.observables, target_energy_base)
        # Compute new energy
        new_obs = self.observables(new_mtx)
        new_ham = erg_energy(new_obs, params)
        # Reverse proposal
        log_q_rev = self._log_q_reverse(new_mtx, mtx, params, self.observables, cache)
        # MH accept
        log_alpha = (new_ham - ham).item() + (log_q_rev - log_q_fwd)
        acc = 1.0 if log_alpha >= 0 else float(torch.exp(torch.tensor(log_alpha)).item())
        return new_mtx, acc
    
    def observables(self, mtx):
        pass




class DLMC_Hybrid_Sampler(BaseSampler):
    """
    DLMC with composite energy: E_total = E_ERG(params) + E_model(mtx).
    Same factorized proposal; all ratios computed under composite energy.
    """
    def __init__(self, backend: str, model: nn.Module, h: float = 1.0, g_kind: str = "sqrt",
                 w: float = 1.0, changed_only: bool = False, local_delta_fn=None):
        super().__init__(backend, model)
        self.h = float(h)
        self.g_kind = g_kind
        self.w = float(w)
        self.changed_only = changed_only
        self.local_delta_fn = local_delta_fn  # optional fast local ΔE callback

    @torch.no_grad()
    def _energy_edge_value(self, mtx, params, observables_fn, i, j, bit):
        if self.local_delta_fn is not None:
            return target_energy_hybrid(mtx, params, observables_fn, self.model) + self.local_delta_fn(mtx, i, j, bit)
        # naive fallback
        old = mtx[i, j].item()
        if int(old) == int(bit):
            return target_energy_hybrid(mtx, params, observables_fn, self.model)
        tmp = mtx.clone()
        tmp[i, j] = bit
        tmp[j, i] = bit
        return target_energy_hybrid(tmp, params, observables_fn, self.model)

    def _Q_rates_binary(self, E0, E1):
        dE = (E1 - E0)
        r10 = torch.exp(-(dE))
        r01 = torch.exp(-(-dE))
        g1 = g_balanced(r10, self.g_kind)
        g0 = g_balanced(r01, self.g_kind)
        alpha_ = self.w * g1
        beta_ = self.w * g0
        return alpha_, beta_

    def _row_Ph_binary(self, cur_bit, alpha_, beta_):
        s = (alpha_ + beta_).clamp_min(1e-30)
        e = torch.exp(-s * self.h)
        p00 = beta_/s + (alpha_/s) * e
        p01 = alpha_/s - (alpha_/s) * e
        p10 = beta_/s - (beta_/s) * e
        p11 = alpha_/s + (beta_/s) * e
        return torch.stack([p00, p01]) if int(cur_bit) == 0 else torch.stack([p10, p11])

    def _build_proposal_and_sample(self, mtx, params, observables_fn):
        n = mtx.shape[0]
        cur = mtx.clone()
        new = mtx.clone()
        log_q_fwd = 0.0
        cache = []
        for i, j in iter_upper_tri_indices(n):
            cur_bit = int(cur[i, j].item())
            E0 = self._energy_edge_value(cur, params, observables_fn, i, j, 0)
            E1 = self._energy_edge_value(cur, params, observables_fn, i, j, 1)
            alpha_, beta_ = self._Q_rates_binary(E0, E1)
            row = self._row_Ph_binary(cur_bit, alpha_, beta_)
            row = torch.clamp(row, 1e-30, 1.0)
            row = row / row.sum()
            new_bit = torch.multinomial(row, 1).item()
            log_q_fwd += torch.log(row[new_bit]).item()
            if new_bit != cur_bit:
                new[i, j] = new_bit
                new[j, i] = new_bit
            cache.append((i, j, cur_bit, new_bit))
        return new, log_q_fwd, cache

    def _log_q_reverse(self, new_mtx, params, observables_fn, cache):
        log_q_rev = 0.0
        for (i, j, cur_bit, new_bit) in cache:
            E0y = self._energy_edge_value(new_mtx, params, observables_fn, i, j, 0)
            E1y = self._energy_edge_value(new_mtx, params, observables_fn, i, j, 1)
            alpha_y, beta_y = self._Q_rates_binary(E0y, E1y)
            row_y = self._row_Ph_binary(int(new_bit), alpha_y, beta_y)
            row_y = torch.clamp(row_y, 1e-30, 1.0)
            row_y = row_y / row_y.sum()
            log_q_rev += torch.log(row_y[int(cur_bit)]).item()
        return log_q_rev

    def proposal(self, mtx, obs, params, ham):
        mtx = mtx.to(self.backend)
        params = params.to(self.backend)
        # Build
        new_mtx, log_q_fwd, cache = self._build_proposal_and_sample(mtx, params, self.observables)
        # Energies
        new_obs = self.observables(new_mtx)
        new_ham = erg_energy(new_obs, params) + self.model(new_mtx).view(()).to(new_obs)
        # Reverse
        log_q_rev = self._log_q_reverse(new_mtx, params, self.observables, cache)
        # MH
        log_alpha = (new_ham - ham).item() + (log_q_rev - log_q_fwd)
        acc = 1.0 if log_alpha >= 0 else float(torch.exp(torch.tensor(log_alpha)).item())
        return new_mtx, acc
    
    def observables(self, mtx):
        pass

def _upper_mask(n, device):
    m = torch.ones((n, n), dtype=torch.bool, device=device).triu(1)
    return m

def _grad_wrt_mtx(scalar, mtx):
    (g,) = torch.autograd.grad(scalar, mtx, retain_graph=False, create_graph=False)
    return g


class DLP_Sampler(BaseSampler):
    """
    Discrete Langevin Proposal (binary, DMALA) for ERG energy U(x)=<g(x),theta>.
    Matches your proposal(...) signature and returns (new_mtx, acceptance_prob).
    """
    def __init__(self, backend: str, stepsize_alpha: float = 0.2):
        super().__init__(backend)
        self.alpha = float(stepsize_alpha)

    def _flip_block(self, x, mask, flips):
        y = x.clone().detach()
        y[mask] = (1. - x[mask]) * flips + x[mask] * (1 - flips)  # xor toggle
        #(1. - x_cur)*ind + x_cur * (1. - ind)
        y = torch.triu(y, 1)
        y = y + y.T
        return y

    def _dlp_probs(self, mtx, ham_scalar, mask):
        """
        Compute per-edge flip probabilities p_i (binary case, Appendix A).
        P(θ)_i = sigmoid(-0.5 * grad_i * (2θ_i - 1) - 1/(2α))
        """
        g = _grad_wrt_mtx(ham_scalar, mtx)               # ∇U(θ) wrt adjacency
        theta = mtx[mask]                                 # binary {0,1}
        sign = (2.0 * theta - 1.0)                        # in {-1, +1}
        gi = g[mask]
        logit = -0.5 * gi * sign - (0.5 / self.alpha)
        p = torch.sigmoid(logit).clamp(1e-12, 1.0 - 1e-12)
        return p

    def proposal(self, mtx, obs, params, ham):
        device = mtx.device
        n = mtx.shape[0]
        mask = _upper_mask(n, device)

        # Ensure fresh graph with grad for current U
        x = mtx.detach().clone().to(device).requires_grad_(True)
        Ux = self._hamiltonian(self.observables(x), params)  # scalar U(x)

        # ---- Forward: factorized flip sampling
        with torch.no_grad():
            p_fwd = self._dlp_probs(x, Ux, mask)            # per-edge flip probs
            flips = torch.bernoulli(p_fwd).to(dtype=torch.int32)
            log_q_fwd = (flips * torch.log(p_fwd) + (~flips) * torch.log(1 - p_fwd)).sum().item()
            y = self._flip_block(x, mask, flips)

        # ---- New energy
        y = y.detach().clone().requires_grad_(True)
        Uy = self._hamiltonian(self.observables(y), params)

        # ---- Reverse proposal probs computed at y
        with torch.no_grad():
            p_rev = self._dlp_probs(y, Uy, mask)
            log_q_rev = (flips * torch.log(p_rev) + (~flips) * torch.log(1 - p_rev)).sum().item()

        # ---- MH acceptance in log-space
        log_alpha = (Uy - Ux).item() + (log_q_rev - log_q_fwd)
        #print(log_alpha, Uy - Ux, log_q_rev, log_q_fwd)
        acc = 1.0 if log_alpha >= 0 else float(torch.exp(torch.tensor(log_alpha)).item())
        return y.detach(), acc

    def observables(self, mtx):
        pass


class DLP_Hybrid_Sampler(BaseSampler):
    """
    Composite U(x) = U_ERG(x;theta) + U_model(x).
    Same DLP (DMALA) logic; gradient is taken wrt the composite U.
    """
    def __init__(self, backend: str, model: nn.Module, stepsize_alpha: float = 0.3):
        super().__init__(backend, model)
        self.alpha = float(stepsize_alpha)
        self.model = model.to(self.backend)

    def _upper_mask(self, n, device):
        return _upper_mask(n, device)

    def _flip_block(self, x, mask, flips):
        return DLP_Sampler._flip_block(self, x, mask, flips)

    def _U_total(self, m, params):
        base = self._hamiltonian(self.observables(m), params)
        model_term = self.model(m)
        if model_term.dim() > 0:  # ensure scalar
            model_term = model_term.view(()).to(base)
        return base + model_term

    def _dlp_probs(self, mtx, U_scalar, mask):
        g = _grad_wrt_mtx(U_scalar, mtx)               # ∇U_total
        theta = mtx[mask]
        sign = (2.0 * theta - 1.0)
        gi = g[mask]
        logit = -0.5 * gi * sign - (0.5 / self.alpha)
        p = torch.sigmoid(logit).clamp(1e-12, 1.0 - 1e-12)
        return p

    def proposal(self, mtx, obs, params, ham):
        device = mtx.device
        n = mtx.shape[0]
        mask = self._upper_mask(n, device)

        x = mtx.detach().clone().to(device).requires_grad_(True)
        Ux = self._U_total(x, params)

        with torch.no_grad():
            p_fwd = self._dlp_probs(x, Ux, mask)
            flips = torch.bernoulli(p_fwd).to(dtype=torch.bool)
            log_q_fwd = (flips * torch.log(p_fwd) + (~flips) * torch.log(1 - p_fwd)).sum().item()
            y = self._flip_block(x, mask, flips)

        y = y.detach().clone().requires_grad_(True)
        Uy = self._U_total(y, params)

        with torch.no_grad():
            p_rev = self._dlp_probs(y, Uy, mask)
            log_q_rev = (flips * torch.log(p_rev) + (~flips) * torch.log(1 - p_rev)).sum().item()

        log_alpha = (Uy - Ux).item() + (log_q_rev - log_q_fwd)
        acc = 1.0 if log_alpha >= 0 else float(torch.exp(torch.tensor(log_alpha)).item())
        return y.detach(), acc

    def observables(self, mtx):
        pass



import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ============================================================
# Low-level utilities
# ============================================================

def upper_triangle_mask(n: int, device: torch.device) -> torch.Tensor:
    """
    Boolean mask selecting strictly upper-triangular entries of an NxN matrix.
    """
    return torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=device),
        diagonal=1
    )


def apply_edge_flips(
    adj: torch.Tensor,
    mask: torch.Tensor,
    flips: torch.Tensor
) -> torch.Tensor:
    """
    Apply edge flips to an undirected adjacency matrix.

    Parameters
    ----------
    adj   : [N, N] float tensor in {0,1}
    mask  : upper-triangular boolean mask
    flips : [E] boolean tensor

    Returns
    -------
    Symmetric adjacency matrix with flipped edges.
    """
    new_adj = adj.clone()

    upper_edges = new_adj[mask].bool()
    new_adj[mask] = (upper_edges ^ flips).to(adj.dtype)

    new_adj = torch.triu(new_adj, diagonal=1)
    new_adj = new_adj + new_adj.T
    new_adj.fill_diagonal_(0.0)

    return new_adj


def grad_wrt_adjacency(
    scalar_energy: torch.Tensor,
    adj: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradient of scalar energy w.r.t. adjacency matrix.
    """
    if scalar_energy.ndim != 0:
        raise ValueError(f"Expected scalar energy, got {scalar_energy.shape}")

    (grad,) = torch.autograd.grad(
        scalar_energy,
        adj,
        retain_graph=False,
        create_graph=False
    )
    return grad


def log_balancing_weight(
    delta: torch.Tensor,
    kind: str = "sqrt"
) -> torch.Tensor:
    """
    Log g(delta) for locally-balanced proposals.

    delta = log π(x^flip) - log π(x)
    """
    if kind == "sqrt":
        return 0.5 * delta
    elif kind in ("ratio", "a_over_1p_a"):
        return F.logsigmoid(delta)
    else:
        raise ValueError(f"Unknown g_kind='{kind}'")


def log1mexp_of_exp(a: torch.Tensor) -> torch.Tensor:
    """
    Stable computation of log(1 - exp(-exp(a))).
    """
    t = torch.exp(a)
    out = torch.empty_like(a)

    large = t > 20.0
    out[large] = 0.0
    out[~large] = torch.log1p(-torch.exp(-t[~large]) + 1e-30)

    return out


# ============================================================
# DLMC state
# ============================================================

@dataclass
class DLMCState:
    steps: int = 0
    log_tau: float = 0.0
    log_z: float = 0.0


# ============================================================
# DLMC Hybrid Sampler
# ============================================================

class DLMC_Hybrid_Sampler:
    """
    Exact DISCS-style Binary DLMC sampler (interpolate solver) for:

        log π(G) = <observables(G), params> + model(G)

    Assumptions:
    - model(adj) returns a scalar
    - observables(adj) is differentiable
    - adj is symmetric, binary, zero diagonal
    """

    def __init__(
        self,
        base_sampler,
        model: nn.Module,
        g_kind: str = "sqrt",
        n_target: float = 3.0,
        adaptive: bool = True,
        target_accept: float = 0.574,
    ):
        """
        Parameters
        ----------
        base_sampler : instance of your BaseSampler (for observables + Hamiltonian)
        model        : deep EBM, called as model(adj)
        """
        self.base = base_sampler
        self.model = model.to(base_sampler.backend)

        self.g_kind = g_kind
        self.n_target = float(n_target)
        self.adaptive = adaptive
        self.target_accept = float(target_accept)

        self.state = DLMCState()

    # --------------------------------------------------------

    def logp(self, adj: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Composite log-density.
        """
        erg_term = self.base._hamiltonian(self.base.observables(adj), params)
        deep_term = self.model(adj)

        if deep_term.numel() != 1:
            raise ValueError("model(adj) must return a scalar")

        return erg_term + deep_term.view(())

    # --------------------------------------------------------

    @torch.no_grad()
    def proposal_distribution(
        self,
        delta: torch.Tensor,
        log_tau: torch.Tensor
    ) -> torch.Tensor:
        """
        DISCS interpolate solver (binary DLMC).
        """
        log_w = log_balancing_weight(delta, self.g_kind)
        log_nu = F.logsigmoid(delta)

        s = log_tau + log_w - log_nu
        threshold = log_nu + log1mexp_of_exp(s)

        return torch.exp(
            torch.clamp(threshold, max=log_nu)
        ).clamp(1e-12, 1.0 - 1e-12)

    # --------------------------------------------------------

    def maybe_update_log_tau(self, log_w: torch.Tensor, acc: float):
        """
        DISCS-style adaptation of tau.
        """
        log_z = torch.logsumexp(log_w, dim=0)

        if self.state.steps == 0:
            log_tau = torch.log(torch.tensor(self.n_target, device=log_w.device)) - log_z
            self.state = DLMCState(steps=1, log_tau=float(log_tau), log_z=float(log_z))
            return

        if not self.adaptive:
            self.state.steps += 1
            self.state.log_z = float(log_z)
            return

        n = torch.exp(torch.tensor(self.state.log_tau + float(log_z)))
        n = n + 3.0 * (acc - self.target_accept)
        n = torch.clamp(n, min=1e-6)

        self.state.log_tau = float(torch.log(n) - log_z)
        self.state.log_z = float(log_z)
        self.state.steps += 1

    # --------------------------------------------------------

    def proposal(
        self,
        graph: torch.Tensor,
        params: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate one DLMC proposal.
        """
        device = torch.device(self.base.backend)
        params = params.to(device)

        x = graph.to(device).detach().clone().requires_grad_(True)
        n = x.shape[0]
        mask = upper_triangle_mask(n, device)

        # ----- forward quantities
        logp_x = self.logp(x, params)
        grad_x = grad_wrt_adjacency(logp_x, x)
        delta_x = (1.0 - 2.0 * x[mask]) * grad_x[mask]
        log_w_x = log_balancing_weight(delta_x, self.g_kind)

        log_tau = torch.tensor(self.state.log_tau, device=device)
        if self.state.steps == 0:
            log_tau = torch.log(torch.tensor(self.n_target, device=device)) - torch.logsumexp(log_w_x, dim=0)

        with torch.no_grad():
            dist_x = self.proposal_distribution(delta_x, log_tau)
            flips = torch.bernoulli(dist_x).bool()
            ll_x2y = (torch.log(dist_x) * flips + torch.log(1 - dist_x) * (~flips)).sum()
            y = apply_edge_flips(x.detach(), mask, flips)

        # ----- reverse quantities
        y_req = y.detach().clone().requires_grad_(True)
        logp_y = self.logp(y_req, params)
        grad_y = grad_wrt_adjacency(logp_y, y_req)
        delta_y = (1.0 - 2.0 * y_req[mask]) * grad_y[mask]

        with torch.no_grad():
            dist_y = self.proposal_distribution(delta_y, log_tau)
            ll_y2x = (torch.log(dist_y) * flips + torch.log(1 - dist_y) * (~flips)).sum()

        # ----- MH acceptance
        log_acc = (logp_y + ll_y2x) - (logp_x + ll_x2y)
        acc = torch.exp(torch.clamp(log_acc, max=0.0)).item()

        self.maybe_update_log_tau(log_w_x.detach(), acc)

        return y.detach(), acc
    def observables(self, mtx):
        pass
