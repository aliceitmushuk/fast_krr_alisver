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
