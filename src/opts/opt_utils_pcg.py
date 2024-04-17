import torch

from ..preconditioners.nystrom import Nystrom
from ..preconditioners.partial_cholesky import PartialCholesky
from ..preconditioners.falkon import Falkon


def _get_precond(model, precond_params, device):
    precond = None
    if precond_params is not None:
        if precond_params["type"] == "nystrom":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }

            K_lin_op = model._get_full_lin_op()

            precond = Nystrom(device, **precond_params_sub)
            precond.update(K_lin_op, model.n)
        elif precond_params["type"] == "partial_cholesky":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }
            precond = PartialCholesky(device, **precond_params_sub)
            precond.update(model.x, model.kernel_params)
    return precond


def _get_precond_inducing(model, precond_params, device):
    if precond_params["type"] == "falkon":
        precond = Falkon(device)
        precond.update(model.K_mm, model.n, model.m, model.lambd)
    return precond


def _init_pcg(w0, K_lin_op, b, precond):
    r = b - K_lin_op(w0)
    z = _apply_precond(r, precond)
    p = z.clone()
    return r, z, p


def _step_pcg(w, r, z, p, K_lin_op, precond):
    Kp = K_lin_op(p)
    r0_dot_z0 = torch.dot(r, z)
    alpha = r0_dot_z0 / torch.dot(p, Kp)
    w += alpha * p
    r -= alpha * Kp
    z = _apply_precond(r, precond)
    beta = torch.dot(r, z) / r0_dot_z0
    p = z + beta * p
    return w, r, z, p


def _apply_precond(v, precond):
    if precond is not None:
        return precond.inv_lin_op(v)
    else:
        return v
