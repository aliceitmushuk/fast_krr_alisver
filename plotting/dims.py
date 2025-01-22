import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pykeops.torch import LazyTensor
import torch

from constants import (
    USE_LATEX,
    FONTSIZE,
    LEGEND_SPECS,
    SZ_COL,
    SZ_ROW,
    BASE_SAVE_DIR,
    EXTENSION,
)
from base_utils import render_in_latex, set_fontsize, get_save_path

SEED = 0
N = 10_000
D = 10
LAMBDA_UNSCALED = 1e-6
GAMMAS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SAVE_DIR = "flat_dim"


def _import_src_utils():
    # Get the current working directory
    original_dir = os.getcwd()

    try:
        # Change to the directory containing the module
        new_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        os.chdir(new_dir)

        # Add the directory to the Python path (if necessary)
        sys.path.append(new_dir)

        # Import the module
        from src.kernels.kernel_inits import _get_kernel
        from src.experiment_utils import set_random_seed

        return _get_kernel, set_random_seed
    finally:
        # Change back to the original directory
        os.chdir(original_dir)


def _get_actual_tensor(lazy_tensor, dim, device):
    return lazy_tensor @ torch.eye(dim, device=device)


def _get_eigs_sorted(mat):
    eigs = torch.linalg.eigvals(mat)
    return torch.sort(eigs.real, descending=True)[0]


def _gamma_lmin_eff_dim(spectrum, gamma, lambd):
    return torch.sum(spectrum / (spectrum + gamma * (lambd + torch.min(spectrum))))


def _gamma_flat_dim(spectrum, gamma, lambd):
    return torch.sum(spectrum > gamma * (lambd + torch.min(spectrum)))


DIM_FNS = {"eff": _gamma_lmin_eff_dim, "flat": _gamma_flat_dim}


def _get_dim(dim_type, spectrum, gamma, lambd):
    dim_fn = DIM_FNS.get(dim_type, None)
    if dim_fn is None:
        raise ValueError(f"Invalid type {dim_type}")
    return dim_fn(spectrum, gamma, lambd)


def _get_dim_multiple(dim_type, spectrum, gammas, lambd):
    return np.array([_get_dim(dim_type, spectrum, gamma, lambd) for gamma in gammas])


if __name__ == "__main__":
    _get_kernel, set_random_seed = _import_src_utils()

    if USE_LATEX:
        render_in_latex()
    set_fontsize(FONTSIZE)
    set_random_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figsize = (2 * SZ_COL, SZ_ROW)
    save_dir = os.path.join(BASE_SAVE_DIR, SAVE_DIR)

    lambd = N * LAMBDA_UNSCALED
    l1laplace_params = {"type": "l1_laplace", "sigma": D**0.5}
    rbf_params = {"type": "rbf", "sigma": D**0.5}
    matern52_params = {"type": "matern", "sigma": D**0.5, "nu": 5 / 2}

    # Compute kernel matrices
    X = torch.randn(N, D, device=device)
    X1_lazy = LazyTensor(X[:, None, :])
    X2_lazy = LazyTensor(X[None, :, :])
    l1laplace_kernel = _get_kernel(X1_lazy, X2_lazy, l1laplace_params)
    rbf_kernel = _get_kernel(X1_lazy, X2_lazy, rbf_params)
    matern52_kernel = _get_kernel(X1_lazy, X2_lazy, matern52_params)
    l1laplace_kernel = _get_actual_tensor(l1laplace_kernel, N, device)
    rbf_kernel = _get_actual_tensor(rbf_kernel, N, device)
    matern52_kernel = _get_actual_tensor(matern52_kernel, N, device)

    # Compute kernel eigenvalues
    l1laplace_eigs = _get_eigs_sorted(l1laplace_kernel.cpu())
    rbf_eigs = _get_eigs_sorted(rbf_kernel.cpu())
    matern52_eigs = _get_eigs_sorted(matern52_kernel.cpu())

    # Compute effective and flat dimensions
    l1_laplace_eff_dims = _get_dim_multiple("eff", l1laplace_eigs, GAMMAS, lambd)
    rbf_eff_dims = _get_dim_multiple("eff", rbf_eigs, GAMMAS, lambd)
    matern52_eff_dims = _get_dim_multiple("eff", matern52_eigs, GAMMAS, lambd)
    l1_laplace_flat_dims = _get_dim_multiple("flat", l1laplace_eigs, GAMMAS, lambd)
    rbf_flat_dims = _get_dim_multiple("flat", rbf_eigs, GAMMAS, lambd)
    matern52_flat_dims = _get_dim_multiple("flat", matern52_eigs, GAMMAS, lambd)

    # Plot spectra, effective, and flat dimensions
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subfigure 1: Kernel eigenvalues
    axes[0].semilogy(l1laplace_eigs.cpu().numpy(), label="Laplacian", color="tab:blue")
    axes[0].semilogy(
        matern52_eigs.cpu().numpy(), label=r"Mat\'{e}rn-5/2", color="tab:orange"
    )
    axes[0].semilogy(rbf_eigs.cpu().numpy(), label="RBF", color="tab:pink")
    axes[0].set_ylabel(r"$\lambda_i(K)$")
    axes[0].set_xlabel(r"$i$")
    axes[0].set_title("Kernel spectra")

    # Subfigure 2: Effective and flat dimensions
    axes[1].semilogy(GAMMAS, l1_laplace_eff_dims, linestyle="dashed", color="tab:blue")
    axes[1].semilogy(GAMMAS, l1_laplace_flat_dims, linestyle="solid", color="tab:blue")
    axes[1].semilogy(GAMMAS, matern52_eff_dims, linestyle="dashed", color="tab:orange")
    axes[1].semilogy(GAMMAS, matern52_flat_dims, linestyle="solid", color="tab:orange")
    axes[1].semilogy(GAMMAS, rbf_eff_dims, linestyle="dashed", color="tab:pink")
    axes[1].semilogy(GAMMAS, rbf_flat_dims, linestyle="solid", color="tab:pink")

    axes[1].semilogx()
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel(r"Dimension")
    axes[1].set_title("Effective and flat dimensions")

    # Custom legend with three lines for kernels and two for dimensions
    legend_elements = [
        Line2D([0], [0], color="tab:blue", label="Laplacian"),
        Line2D([0], [0], color="tab:orange", label=r"Mat\'{e}rn-5/2"),
        Line2D([0], [0], color="tab:pink", label="RBF"),
        Line2D(
            [0],
            [0],
            linestyle="dashed",
            color="black",
            label=r"$d^{\gamma \lambda}(K)$",
        ),
        Line2D(
            [0],
            [0],
            linestyle="solid",
            color="black",
            label=r"$d_\flat^\gamma(K_{\lambda})$",
        ),
    ]
    fig.legend(handles=legend_elements, **LEGEND_SPECS)

    # Save the figure
    fig.tight_layout()
    plt.savefig(
        get_save_path(save_dir, f"spectra_dim.{EXTENSION}"), bbox_inches="tight"
    )
