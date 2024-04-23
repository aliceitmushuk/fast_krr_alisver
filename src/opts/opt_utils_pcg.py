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

            K_lin_op, K_trace = model._get_full_lin_op()

            precond = Nystrom(device, **precond_params_sub)
            precond.update(K_lin_op, K_trace, model.n)
        elif precond_params["type"] == "partial_cholesky":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }
            precond = PartialCholesky(device, **precond_params_sub)
            diag_K = model._get_diag()
            precond.update(model.x, model.kernel_params, model.K, diag_K)
    return precond


def _get_precond_inducing(model, precond_params, device):
    if precond_params["type"] == "falkon":
        precond = Falkon(device)
        precond.update(model.K_mm, model.n, model.m, model.lambd)
    return precond
