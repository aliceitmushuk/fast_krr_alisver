from .opt_utils import _get_L
from ..preconditioners.nystrom import Nystrom


def _get_precond_L(model, bH, bH2, precond_params):
    (
        subsampled_lin_op,
        subsampled_reg_lin_op,
        subsampled_trace,
    ) = model._get_subsampled_lin_ops(bH, bH2)

    precond = None

    if precond_params is not None:
        if precond_params["type"] == "nystrom":
            precond_params_sub = {
                k: v for k, v in precond_params.items() if k != "type"
            }
            precond = Nystrom(model.device, **precond_params_sub)
            precond.update(subsampled_lin_op, subsampled_trace, model.m)
            L = _get_L(subsampled_reg_lin_op, precond.inv_lin_op, model.m, model.device)
    else:  # No preconditioner
        L = _get_L(subsampled_reg_lin_op, lambda x: x, model.m, model.device)

    return precond, L


def _get_minibatch(generator):
    try:
        idx = next(generator)
        return idx
    except StopIteration:
        return None
