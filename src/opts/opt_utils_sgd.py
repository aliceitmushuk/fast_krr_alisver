import numpy as np
import torch
from pykeops.torch import LazyTensor

from .opt_utils import _get_L
from ..preconditioners.nystrom import Nystrom
from ..kernels.kernel_inits import _get_kernel

def _get_precond_L_inducing(model, bH, precond_params):
    hess_pts = torch.from_numpy(np.random.choice(model.n, bH, replace=False))
    x_hess_i = LazyTensor(model.x[hess_pts][:, None, :])
    K_sm = _get_kernel(x_hess_i, model.x_inducing_j, model.kernel_params)

    hess_pts_lr = torch.from_numpy(np.random.choice(model.n, bH, replace=False))
    x_hess_lr_i = LazyTensor(model.x[hess_pts_lr][:, None, :])
    K_sm_lr = _get_kernel(x_hess_lr_i, model.x_inducing_j, model.kernel_params)

    adj_factor = model.n / bH

    def K_inducing_sub_lin_op(v):
        return adj_factor * K_sm.T @ (K_sm @ v)

    def K_inducing_sub_Kmm_lin_op(v):
        return adj_factor * K_sm_lr.T @ (K_sm_lr @ v) + model.lambd * (model.K_mm @ v)

    precond = None

    if precond_params is not None:
        if precond_params["type"] == "nystrom":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }
            precond = Nystrom(model.device, **precond_params_sub)
            precond.update(K_inducing_sub_lin_op, model.m)
            L = _get_L(
                K_inducing_sub_Kmm_lin_op, precond.inv_sqrt_lin_op, model.m, model.device
            )
    else:  # No preconditioner
        L = _get_L(K_inducing_sub_Kmm_lin_op, lambda x: x, model.m, model.device)

    return precond, L


def _get_stochastic_grad_inducing(model, idx, w):
    x_idx_i = LazyTensor(model.x[idx][:, None, :])
    K_nm_idx = _get_kernel(x_idx_i, model.x_inducing_j, model.kernel_params)
    g = model.n / idx.shape[0] * (K_nm_idx.T @ (K_nm_idx @ w - model.b[idx])) + model.lambd * (model.K_mm @ w)

    return g


# NOTE: This works because of the structure of the KRR objective -- does not work for general objectives
def _get_stochastic_grad_diff_inducing(model, idx, w1, w2):
    x_idx_i = LazyTensor(model.x[idx][:, None, :])
    K_nm_idx = _get_kernel(x_idx_i, model.x_inducing_j, model.kernel_params)
    w_diff = w1 - w2
    g_diff = model.n / idx.shape[0] * (K_nm_idx.T @ (K_nm_idx @ w_diff)) + model.lambd * (
        model.K_mm @ w_diff
    )

    return g_diff


def _get_full_grad_inducing(model, w):
    return model.K_nm.T @ (model.K_nm @ w - model.b) + model.lambd * (model.K_mm @ w)


def _get_table_aux(model, idx, w, table):
    x_idx_i = LazyTensor(model.x[idx][:, None, :])
    K_nm_idx = _get_kernel(x_idx_i, model.x_inducing_j, model.kernel_params)
    new_weights = model.n * (K_nm_idx @ w - model.b[idx])
    aux = K_nm_idx.T @ (new_weights - table[idx])
    return new_weights, aux


def _apply_precond(v, precond):
    if precond is not None:
        return precond.inv_lin_op(v)
    else:
        return v


def _get_minibatch(generator):
    try:
        idx = next(generator)
        return idx
    except StopIteration:
        return None
