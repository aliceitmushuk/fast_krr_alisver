import os
import sys

import matplotlib.pyplot as plt
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
N = 10000
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


def _gamma_lmin_eff_dim(spectrum, gamma, lambd):
    return torch.sum(spectrum / (spectrum + gamma * (lambd + torch.min(spectrum))))


def _gamma_flat_dim(spectrum, gamma, lambd):
    return torch.sum(spectrum > gamma * (lambd + torch.min(spectrum)))


if __name__ == "__main__":
    _get_kernel, set_random_seed = _import_src_utils()

    if USE_LATEX:
        render_in_latex()
    set_fontsize(FONTSIZE)
    set_random_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figsize = (SZ_COL, SZ_ROW)
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
    l1laplace_eigs = torch.linalg.eigvals(l1laplace_kernel.cpu())
    rbf_eigs = torch.linalg.eigvals(rbf_kernel.cpu())
    matern52_eigs = torch.linalg.eigvals(matern52_kernel.cpu())
    l1laplace_eigs = torch.sort(l1laplace_eigs.real, descending=True)[0]
    rbf_eigs = torch.sort(rbf_eigs.real, descending=True)[0]
    matern52_eigs = torch.sort(matern52_eigs.real, descending=True)[0]

    # Plot kernel eigenvalues
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.semilogy(l1laplace_eigs.cpu().numpy(), label="Laplacian", color="tab:blue")
    axes.semilogy(matern52_eigs.cpu().numpy(), label="Matern 5/2", color="tab:orange")
    axes.semilogy(rbf_eigs.cpu().numpy(), label="RBF", color="tab:pink")
    axes.set_ylabel(r"$\lambda_i(K)$")
    axes.set_xlabel(r"$i$")
    fig.legend(**LEGEND_SPECS)
    fig.tight_layout()
    plt.savefig(get_save_path(save_dir, f"spectra.{EXTENSION}"), bbox_inches="tight")

    # Compute effective and flat dimensions
    l1_laplace_eff_dims = np.array(
        [_gamma_lmin_eff_dim(l1laplace_eigs, gamma, lambd) for gamma in GAMMAS]
    )
    rbf_eff_dims = np.array(
        [_gamma_lmin_eff_dim(rbf_eigs, gamma, lambd) for gamma in GAMMAS]
    )
    matern52_eff_dims = np.array(
        [_gamma_lmin_eff_dim(matern52_eigs, gamma, lambd) for gamma in GAMMAS]
    )

    l1_laplace_flat_dims = np.array(
        [_gamma_flat_dim(l1laplace_eigs, gamma, lambd) for gamma in GAMMAS]
    )
    rbf_flat_dims = np.array(
        [_gamma_flat_dim(rbf_eigs, gamma, lambd) for gamma in GAMMAS]
    )
    matern52_flat_dims = np.array(
        [_gamma_flat_dim(matern52_eigs, gamma, lambd) for gamma in GAMMAS]
    )

    # Plot effective and flat dimensions
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.semilogy(
        GAMMAS,
        l1_laplace_eff_dims,
        label=r"Laplacian $d^{\gamma \lambda}(K)$",
        linestyle="dashed",
        color="tab:blue",
    )
    axes.semilogy(
        GAMMAS,
        l1_laplace_flat_dims,
        label=r"Laplacian $d_\flat^\gamma(K_{\lambda})$",
        linestyle="solid",
        color="tab:blue",
    )
    axes.semilogy(
        GAMMAS,
        matern52_eff_dims,
        label=r"Matern 5/2 $d^{\gamma \lambda}(K)$",
        linestyle="dashed",
        color="tab:orange",
    )
    axes.semilogy(
        GAMMAS,
        matern52_flat_dims,
        label=r"Matern 5/2 $d_\flat^\gamma(K_{\lambda})$",
        linestyle="solid",
        color="tab:orange",
    )
    axes.semilogy(
        GAMMAS,
        rbf_eff_dims,
        label=r"RBF $d^{\gamma \lambda}(K)$",
        linestyle="dashed",
        color="tab:pink",
    )
    axes.semilogy(
        GAMMAS,
        rbf_flat_dims,
        label=r"RBF $d_\flat^\gamma(K_{\lambda})$",
        linestyle="solid",
        color="tab:pink",
    )
    axes.semilogx()
    axes.set_xlabel(r"$\gamma$")
    axes.set_ylabel(r"Dimension")
    fig.legend(**LEGEND_SPECS)
    fig.tight_layout()
    plt.savefig(get_save_path(save_dir, f"dimensions.{EXTENSION}"), bbox_inches="tight")

    # Plot ratio of flat to effective dimensions
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.semilogy(
        GAMMAS,
        l1_laplace_flat_dims / l1_laplace_eff_dims,
        label="Laplacian",
        linestyle="solid",
        color="tab:blue",
    )
    axes.semilogy(
        GAMMAS,
        matern52_flat_dims / matern52_eff_dims,
        label="Matern 5/2",
        linestyle="solid",
        color="tab:orange",
    )
    axes.semilogy(
        GAMMAS,
        rbf_flat_dims / rbf_eff_dims,
        label="RBF",
        linestyle="solid",
        color="tab:pink",
    )
    axes.semilogx()
    axes.set_xlabel(r"$\gamma$")
    axes.set_ylabel(r"$d_\flat^\gamma(K_{\lambda}) / d^{\gamma \lambda}(K)$")
    fig.legend(**LEGEND_SPECS)
    fig.tight_layout()
    plt.savefig(
        get_save_path(save_dir, f"dimension_ratio.{EXTENSION}"), bbox_inches="tight"
    )
