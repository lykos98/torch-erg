import os
import time
import pickle
import numpy as np
import torch
from tqdm import tqdm

from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.samplers import (
    MHSampler_Hard,
    GWGSampler,
    DLMC_Sampler,
    DLP_Sampler,
)

#############################################################
#  Observables
#############################################################

def basic_observables(mtx: torch.Tensor) -> torch.Tensor:
    edges = torch.sum(mtx) / 2
    triangles = torch.trace(mtx @ mtx @ mtx) / 6
    return torch.stack([edges, triangles])

#############################################################
# Evaluation function for param_run (EE)
#############################################################

def evaluate_param_run(sampler, graph, obs, params, niter=20000,
                       params_update_every=3, save_every=50,
                       alpha=0.001, min_change=0.005):

    t0 = time.time()

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
        verbose_level=0
    )

    elapsed = time.time() - t0
    final_params = params_hist[-1]

    # sampler.param_run already prints accepted/rejected
    # but we try to extract them from attributes if present
    accepted = getattr(sampler, "accepted_steps", None)
    rejected = getattr(sampler, "rejected_samples", None)
    acc_rate = (accepted / niter) if accepted is not None else None

    return {
        "final_params": final_params.detach().cpu(),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": acc_rate,
        "elapsed_time": elapsed,
        "params_history": params_hist,
        "graphs_history": graph_hist
    }

#############################################################
# Sampler registry
#############################################################

def make_sampler(name: str, backend: str):
    if name == "MH_":
        class S(MHSampler_Hard):
            def __init__(self, backend): super().__init__(backend)
            def observables(self, mtx): return basic_observables(mtx)
        return S(backend)

    elif name == "GWG_":
        class S(GWGSampler):
            def __init__(self, backend): super().__init__(backend)
            def observables(self, mtx): return basic_observables(mtx)
        return S(backend)

    elif name == "DLMC_":
        class S(DLMC_Sampler):
            def __init__(self, backend): super().__init__(backend, h=0.8, g_kind="sqrt", w=1.0)
            def observables(self, mtx): return basic_observables(mtx)
        return S(backend)

    elif name == "DLP_":
        class S(DLP_Sampler):
            def __init__(self, backend): super().__init__(backend, stepsize_alpha=0.3)
            def observables(self, mtx): return basic_observables(mtx)
        return S(backend)
    else:
        raise ValueError(f"Unknown sampler: {name}")

#############################################################
# Main comparison pipeline
#############################################################

def compare_samplers(
        grid_name="30_ieee",
        sampler_list=["MH_", "GWG_", "DLMC_", "DLP_"],
        niter=20000,
        backend="cuda",
        outdir="results/sampler_comparison"
    ):

    os.makedirs(outdir, exist_ok=True)

    # Load graph
    ordmat, ordlist, buslist, countlist = lp.pow_parser(grid_name)
    graph = torch.tensor(ordmat, dtype=torch.float32)

    # Initial parameters for EE
    betas0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    obs0 = basic_observables(graph)

    results = {}

    for sname in sampler_list:
        print(f"\n==============================")
        print(f" Running sampler: {sname}")
        print(f"==============================")

        sampler = make_sampler(sname, backend)
        sampler.accepted_steps = 0
        sampler.rejected_samples = 0

        r = evaluate_param_run(
            sampler=sampler,
            graph=graph,
            observables=obs0,
            params=betas0.clone(),
            niter=niter,
            params_update_every=3,
            save_every=250,
            alpha=0.001,
            min_change=0.005
        )
        results[sname] = r

        print(f"Final parameters: {r['final_params']}")
        print(f"Accepted: {r['accepted']}, Rejected: {r['rejected']}")
        print(f"Acc rate: {r['acceptance_rate']}")
        print(f"Time: {r['elapsed_time']:.3f} sec")

        # save each sampler result separately
        with open(os.path.join(outdir, f"{sname}_{grid_name}_eval.pkl"), "wb") as f:
            pickle.dump(r, f)

    # Save combined summary
    summary_path = os.path.join(outdir, f"summary_{grid_name}.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(results, f)

    print("\nComparison complete!")
    print(f"Saved summary → {summary_path}")
    return results


#############################################################
# Run if main
#############################################################

if __name__ == "__main__":
    compare_samplers(
        grid_name="30_ieee",
        sampler_list=["MH_", "GWG_", "DLMC_", "DLP_" ],
        niter=30000,
        backend="cuda"
    )
