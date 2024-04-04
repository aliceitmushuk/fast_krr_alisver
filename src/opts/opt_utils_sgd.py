import numpy as np
import torch
from pykeops.torch import LazyTensor

from .opt_utils import _get_L
from ..preconditioners.nystrom import Nystrom
from ..kernels import _get_kernel

def _get_needed_quantities_inducing(x, x_tst, inducing_pts, kernel_params, b):
    # Get inducing points kernel
    x_inducing_i = LazyTensor(x[inducing_pts][:, None, :])
    x_inducing_j = LazyTensor(x[inducing_pts][None, :, :])
    K_mm = _get_kernel(x_inducing_i, x_inducing_j, kernel_params)

    # Get kernel between full training set and inducing points
    x_i = LazyTensor(x[:, None, :])
    K_nm = _get_kernel(x_i, x_inducing_j, kernel_params)

    # Get kernel for test set
    x_tst_i = LazyTensor(x_tst[:, None, :])
    K_tst = _get_kernel(x_tst_i, x_inducing_j, kernel_params)

    return x_inducing_j, K_mm, K_nm, K_tst, inducing_pts.shape[0], x.shape[0], torch.norm(b)

def _get_precond_L_inducing(x, m, n, bH, x_inducing_j, kernel_params, K_mm, lambd, precond_params, device):
    hess_pts = torch.from_numpy(np.random.choice(n, bH, replace=False))
    x_hess_i = LazyTensor(x[hess_pts][:, None, :])
    K_sm = _get_kernel(x_hess_i, x_inducing_j, kernel_params)

    hess_pts_lr = torch.from_numpy(np.random.choice(n, bH, replace=False))
    x_hess_lr_i = LazyTensor(x[hess_pts_lr][:, None, :])
    K_sm_lr = _get_kernel(x_hess_lr_i, x_inducing_j, kernel_params)

    adj_factor = n/bH
    def K_inducing_sub_lin_op(v): return adj_factor * K_sm.T @ (K_sm @ v)
    def K_inducing_sub_Kmm_lin_op(v): return adj_factor * K_sm_lr.T @ (K_sm_lr @ v) + lambd * (K_mm @ v)

    precond = None

    if precond_params is not None:
        if precond_params['type'] == 'nystrom':
            precond_params_sub = {k: v for k, v in precond_params.items() if k != 'type'}
            precond = Nystrom(device, **precond_params_sub)
            precond.update(K_inducing_sub_lin_op, m)
            L = _get_L(K_inducing_sub_Kmm_lin_op, lambd, precond.inv_sqrt_lin_op, m, device)
    else: # No preconditioner
        L = _get_L(K_inducing_sub_Kmm_lin_op, lambd, lambda x: x, m, device)

    return precond, L

def _get_stochastic_grad_inducing(x, n, idx, x_inducing_j, kernel_params, K_mm, a, b, lambd):
    x_idx_i = LazyTensor(x[idx][:, None, :])
    K_nm_idx = _get_kernel(x_idx_i, x_inducing_j, kernel_params)
    g = n/idx.shape[0] * (K_nm_idx.T @ (K_nm_idx @ a - b[idx])) + lambd * (K_mm @ a)

    return g

# NOTE: This works because of the structure of the KRR objective -- does not work for general objectives
def _get_stochastic_grad_diff_inducing(x, n, idx, x_inducing_j, kernel_params, K_mm, a, a_tilde, b, lambd):
    x_idx_i = LazyTensor(x[idx][:, None, :])
    K_nm_idx = _get_kernel(x_idx_i, x_inducing_j, kernel_params)
    a_diff = a - a_tilde
    g_diff = n/idx.shape[0] * (K_nm_idx.T @ (K_nm_idx @ a_diff)) + lambd * (K_mm @ a_diff)

    return g_diff

def _get_full_grad_inducing(K_nm, K_mm, a, b, lambd):
    return K_nm.T @ (K_nm @ a - b) + lambd * (K_mm @ a)

def _apply_precond(v, precond):
    if precond is not None:
        return precond.inv_lin_op(v)
    else:
        return v