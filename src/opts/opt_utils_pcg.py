from ..preconditioners.nystrom import Nystrom
from ..preconditioners.partial_cholesky import PartialCholesky
from ..preconditioners.falkon import Falkon


def _get_precond_full(model, precond_params, device):
    precond = None
    if precond_params is not None:
        precond_params_sub = {
            k: v for k, v in precond_params.items() if k != "type" and k != "blk_size"
        }
        if precond_params["type"] == "nystrom":
            K_lin_op, K_trace = model._get_full_lin_op()
            precond = Nystrom(device, **precond_params_sub)
            precond.update(K_lin_op, K_trace, model.n)
        elif precond_params["type"] == "partial_cholesky":
            K_fn = model._get_kernel_fn()
            K_diag = model._get_diag()
            blk_size = precond_params.get("blk_size", None)
            precond = PartialCholesky(device, **precond_params_sub)
            precond.update(K_fn, K_diag, model.x, blk_size)
    return precond


def _get_precond_inducing(model, precond_params, device):
    if precond_params["type"] == "falkon":
        K_mm_lin_op = model._Kmm_lin_op
        K_mm_trace = model._get_Kmm_trace()
        precond = Falkon(device)
        precond.update(K_mm_lin_op, K_mm_trace, model.n, model.m, model.lambd)
    return precond
