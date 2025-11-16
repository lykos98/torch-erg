import torch
import numpy as np

from .utils import *
from tqdm import tqdm
from typing import Union, Tuple
import torch.nn as nn
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from abc import ABC, abstractmethod

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
    
    def param_run(self,   graph:               torch.tensor, 
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
            new_graph, acceptance_prob = self.proposal(current_graph, current_obs, current_params, current_hamiltonian)


            p = torch.rand(1).item()

            if (p < acceptance_prob):

                new_graph.requires_grad_()

                current_graph = new_graph
                current_obs = self.observables(new_graph)
                current_hamiltonian = self._hamiltonian(current_obs, current_params)
                #print("current obs are:", current_obs)

                accepted_steps += 1
                if accepted_steps % params_update_every == 0:
                    update_steps += 1
                    current_params = self._update_parameters(current_obs, start_obs, current_params, alpha, min_change)
                 #   print("params now are: ",current_params)
            else:
                rejected_samples += 1
            if accepted_steps % save_every == 0:
                #print('number of effective updates is: ', update_steps)
                if save_graph:
                    return_graph.append(current_graph.clone().detach())
                if save_params:
                    return_params.append(current_params.clone().detach())

        #some logging info also here fraction of samples accepted
        print('number of accepted steps is: ', accepted_steps)
        print('number of rejected samples: ', rejected_samples)
        print('number of effective updates is: ', update_steps)
        
        return_params.append(current_params.clone().detach())
        return_graph.append(current_graph.clone().detach())
        return return_params, return_graph
    
    def sample_run(self, graph:               torch.tensor, 
                         observables:         torch.tensor, 
                         params:              torch.tensor,
                         niter:               int, 
                         save_every:          int,
                         burn_in:             float = 0.,
                         verbose_level:       int = 0
            ) -> list[torch.tensor]:

        assert burn_in >= 0. and burn_in < 1., "Invalid burn in fraction, should be [0., 1.)"

        burn_in_iter = int(burn_in * niter)
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
    def __init__(self, backend: str, model: nn.Module):
        super().__init__(backend, model)
        self.model = model

        self.model.to(self.backend)

    def _d(self,x):
        with torch.no_grad():
            d = -(2*x - 1) * x.grad
        return d
    def proposal(self,mtx, obs, params, ham):
        
        mtx= mtx.to(self.backend)
        ham = ham.to(self.backend)
        comp_ham = ham + self.model(mtx)
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

        new_ham.backward(retain_graph = True)

        dx = self._d(new_mtx)
        torch.diagonal(dx, 0).zero_()
        with torch.no_grad():
            q_ix_prime = torch.nn.functional.softmax(dx.ravel(), dim=0)

        qq = torch.exp(new_ham - comp_ham) * q_ix_prime[i * mtx.shape[1] + j]/q_ix[i * mtx.shape[1] + j]
        #print("new ham is: ", new_ham)
        #print("old ham is: ", ham)
        #print("qq is: ", qq)
        acceptance_prob = min(1, qq.item())   
        
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

    @torch.no_grad()
    def _energy_edge_value(self, mtx, params, observables_fn, i, j, bit, base_energy_fn):
        # Returns energy when edge (i,j) is set to 'bit' (0/1).
        # Fast path via user callback; fallback to flip-and-eval.
        if self.local_delta_fn is not None:
            return base_energy_fn(mtx, params, observables_fn) + self.local_delta_fn(mtx, i, j, bit)
        # naive fallback
        old = mtx[i, j].item()
        if int(old) == int(bit):
            return base_energy_fn(mtx, params, observables_fn)
        tmp = mtx.clone()
        tmp[i, j] = bit
        tmp[j, i] = bit
        return base_energy_fn(tmp, params, observables_fn)

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

    def _row_Ph_binary(self, cur_bit, alpha_, beta_):
        """
        Exact 2x2 transition (Eq. 37): P^h = exp(Q h) closed-form.
        Return the row corresponding to current bit (0 or 1).
        """
        s = (alpha_ + beta_).clamp_min(1e-30)
        e = torch.exp(-s * self.h)
        # matrix entries
        p00 = beta_/s + (alpha_/s) * e
        p01 = alpha_/s - (alpha_/s) * e
        p10 = beta_/s - (beta_/s) * e
        p11 = alpha_/s + (beta_/s) * e
        if int(cur_bit) == 0:
            return torch.stack([p00, p01])  # probs to [0,1]
        else:
            return torch.stack([p10, p11])

    def _build_proposal_and_sample(self, mtx, params, observables_fn, base_energy_fn):
        """
        Build factorized proposal rows for every edge and sample a full candidate matrix y.
        Returns:
          new_mtx, log_q_fwd (sum over sites), cache with per-site info to compute reverse probs.
        """
        n = mtx.shape[0]
        cur = mtx.clone()
        new = mtx.clone()
        log_q_fwd = 0.0
        # cache info needed for reverse probabilities
        site_cache = []  # tuples: (i,j, cur_bit, new_bit, alpha_, beta_)
        # (Optional) precompute current base energy once
        # We still need E0/E1 per edge; see _energy_edge_value.
        for i, j in iter_upper_tri_indices(n):
            cur_bit = int(cur[i, j].item())
            # energies for bit=0/1 under base target
            E0 = self._energy_edge_value(cur, params, self.observables, i, j, 0, target_energy_base)
            E1 = self._energy_edge_value(cur, params, self.observables, i, j, 1, target_energy_base)
            alpha_, beta_ = self._Q_rates_binary(E0, E1)
            row = self._row_Ph_binary(cur_bit, alpha_, beta_)
            row = torch.clamp(row, 1e-30, 1.0)
            row = row / row.sum()
            new_bit = torch.multinomial(row, 1).item()
            log_q_fwd += torch.log(row[new_bit]).item()
            if new_bit != cur_bit:
                new[i, j] = new_bit
                new[j, i] = new_bit
            site_cache.append((i, j, cur_bit, new_bit, alpha_.item(), beta_.item()))
        return new, log_q_fwd, site_cache

    def _log_q_reverse(self, new_mtx, old_mtx, params, observables_fn, site_cache):
        """
        Compute sum log P^h_n(y_n -> x_n) under state y (reverse rows depend on y context).
        """
        log_q_rev = 0.0
        for (i, j, cur_bit, new_bit, _, _) in site_cache:
            # Under y context, recompute alpha_,beta_
            E0y = self._energy_edge_value(new_mtx, params, observables_fn, i, j, 0, target_energy_base)
            E1y = self._energy_edge_value(new_mtx, params, observables_fn, i, j, 1, target_energy_base)
            alpha_y, beta_y = self._Q_rates_binary(E0y, E1y)
            # row for current (which is new_bit) and prob to go back to cur_bit
            row_y = self._row_Ph_binary(int(new_bit), alpha_y, beta_y)
            row_y = torch.clamp(row_y, 1e-30, 1.0)
            row_y = row_y / row_y.sum()
            log_q_rev += torch.log(row_y[int(cur_bit)]).item()
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
    def __init__(self, backend: str, stepsize_alpha: float = 0.3):
        super().__init__(backend)
        self.alpha = float(stepsize_alpha)

    def _flip_block(self, x, mask, flips):
        y = x.clone()
        # write flips (0/1) into upper triangle then mirror
        y[mask] = (x[mask] ^ flips)  # xor toggle
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
            flips = torch.bernoulli(p_fwd).to(dtype=torch.bool)
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

