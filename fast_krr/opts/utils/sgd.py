from .general import _get_L
from ...preconditioners import preconditioner_inits as pi


def _get_precond_L(model, precond_params, bH, bH2):
    (
        subsampled_lin_op,
        subsampled_lr_lin_op,
        subsampled_trace,
    ) = model._get_subsampled_lin_ops(bH, bH2)

    update_params = None
    if precond_params is not None:
        type = precond_params["type"]
        if type == "newton":
            update_params = {"K_lin_op": subsampled_lr_lin_op, "n": model.m}
        elif type == "nystrom":
            update_params = {
                "K_lin_op": subsampled_lin_op,
                "K_trace": subsampled_trace,
                "n": model.m,
            }
    precond = pi._get_precond(precond_params, update_params, model.lambd, model.device)
    L = _get_L(subsampled_lr_lin_op, precond, model.m, model.device)

    return precond, L


def _get_minibatch(generator):
    try:
        idx = next(generator)
        return idx
    except StopIteration:
        return None
