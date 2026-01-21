import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import torch
from typing import List




def plot_degree_ccdf(
    graphs_real,
    graphs_gen,
    graphs_steered=None,
    outdir=".",
    filename="degree_ccdf.png",
    title="Degree CCDF (log–log)",
    dpi=200,
):
    def ccdf_from_graphs(graphs):
        degrees = []
        for G in graphs:
            degrees.extend([d for _, d in G.degree()])
        degrees = np.array(degrees)
        degrees = degrees[degrees > 0]

        values, counts = np.unique(degrees, return_counts=True)
        pdf = counts / counts.sum()
        ccdf = 1.0 - np.cumsum(pdf)
        return values, ccdf

    plt.figure(figsize=(6, 5))

    x, y = ccdf_from_graphs(graphs_real)
    plt.loglog(x, y, "o-", label="Real", alpha=0.9)

    x, y = ccdf_from_graphs(graphs_gen)
    plt.loglog(x, y, "s-", label="Generated", alpha=0.9)

    if graphs_steered is not None:
        x, y = ccdf_from_graphs(graphs_steered)
        plt.loglog(x, y, "^-", label="Steered", alpha=0.9)

    plt.xlabel("Degree $k$")
    plt.ylabel("CCDF $P(K \\ge k)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=dpi)
    plt.close()



def plot_degree_histogram(
    graphs_real,
    graphs_gen,
    graphs_steered=None,
    outdir=".",
    filename="degree_hist.png",
    bins=20,
    dpi=200,
):
    def collect_deg(graphs):
        return np.array([d for G in graphs for _, d in G.degree()])

    deg_real = collect_deg(graphs_real)
    deg_gen = collect_deg(graphs_gen)

    plt.figure(figsize=(6, 4))
    plt.hist(deg_real, bins=bins, density=True, alpha=0.5, label="Real")
    plt.hist(deg_gen, bins=bins, density=True, alpha=0.5, label="Generated")

    if graphs_steered is not None:
        deg_steered = collect_deg(graphs_steered)
        plt.hist(deg_steered, bins=bins, density=True, alpha=0.5, label="Steered")

    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=dpi)
    plt.close()



def plot_observable_trace(
    obs_list,
    labels,
    outdir=".",
    filename="observable_trace.png",
    burn_in=None,
    dpi=200,
):
    obs = np.array(obs_list)

    if burn_in is not None:
        start = int(burn_in * len(obs))
        obs = obs[start:]

    plt.figure(figsize=(7, 4))
    for i, lab in enumerate(labels):
        plt.plot(obs[:, i], label=lab)

    plt.xlabel("Sampling step")
    plt.ylabel("Observable value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=dpi)
    plt.close()


def plot_observable_bar(
    obs_real,
    obs_gen,
    obs_steered,
    labels,
    outdir=".",
    filename="observable_bar.png",
    dpi=200,
):
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(7, 4))
    plt.bar(x - width, obs_real, width, label="Real")
    plt.bar(x, obs_gen, width, label="Generated")
    plt.bar(x + width, obs_steered, width, label="Steered")

    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=dpi)
    plt.close()


def plot_energy_comparison(
    pkl_path_a: str,
    pkl_path_b: str,
    model: torch.nn.Module,
    device: torch.device,
    node_feat_dim: int = 1,
    label_a: str = "Generated",
    label_b: str = "Reference",
    outpath: str = "energy_comparison.pdf",
    style: str = "kde",  # "kde", "hist", "scatter"
    title: str = "Energy Comparison",
    bins: int = 30,
    dpi: int = 300,
):
    """
    Compare energy distributions assigned by an EBM to two sets of graphs.

    Args:
        pkl_path_a, pkl_path_b: paths to pkl files containing List[nx.Graph]
        model: pretrained EBM (returns scalar energy)
        device: torch device
        node_feat_dim: dimension of constant node features
        label_a, label_b: legend labels
        outpath: output pdf/png path
        style: "kde", "hist", or "scatter"
    """

    model.eval()

    # -------------------------------------------------
    # Load graphs
    # -------------------------------------------------
    with open(pkl_path_a, "rb") as f:
        graphs_a: List[nx.Graph] = pickle.load(f)

    with open(pkl_path_b, "rb") as f:
        graphs_b: List[nx.Graph] = pickle.load(f)

    # -------------------------------------------------
    # Energy evaluation
    # -------------------------------------------------
    def energies_from_graphs(graphs):
        energies = []
        with torch.no_grad():
            for G in graphs:
                A = torch.tensor(
                    nx.to_numpy_array(G),
                    dtype=torch.float32,
                    device=device,
                )
                n = A.size(0)
                feats = torch.ones((n, node_feat_dim), device=device)
                E = model(feats, A)
                energies.append(E.item())
        return np.array(energies)

    E_a = energies_from_graphs(graphs_a)
    E_b = energies_from_graphs(graphs_b)

    # -------------------------------------------------
    # Plotting (paper typography)
    # -------------------------------------------------
    plt.rcParams.update({
        #"font.family": "serif",
        #"font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    plt.figure(figsize=(6.0, 4.2))

    if style == "hist":
        plt.hist(E_a, bins=bins, density=True, alpha=0.55, label=label_a)
        plt.hist(E_b, bins=bins, density=True, alpha=0.55, label=label_b)

    elif style == "kde":
        from scipy.stats import gaussian_kde

        kde_a = gaussian_kde(E_a)
        kde_b = gaussian_kde(E_b)

        xs = np.linspace(
            min(E_a.min(), E_b.min()),
            max(E_a.max(), E_b.max()),
            400,
        )

        plt.plot(xs, kde_a(xs), lw=2.2, label=label_a)
        plt.plot(xs, kde_b(xs), lw=2.2, label=label_b)

    elif style == "scatter":
        plt.scatter(
            np.arange(len(E_a)),
            E_a,
            alpha=0.7,
            s=20,
            label=label_a,
        )
        plt.scatter(
            np.arange(len(E_b)),
            E_b,
            alpha=0.7,
            s=20,
            label=label_b,
        )
        plt.xlabel("Sample index")

    else:
        raise ValueError(f"Unknown style: {style}")

    plt.xlabel("Energy")
    plt.ylabel("Density" if style != "scatter" else "Energy")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3, linestyle="--")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()

    print(f"[INFO] Saved energy comparison plot to {outpath}")
