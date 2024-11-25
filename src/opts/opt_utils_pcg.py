from ..preconditioners import preconditioner_inits as pi


def _get_precond(model, precond_params):
    update_params = None
    if precond_params is not None:
        type = precond_params["type"]
        if type == "falkon":
            K_mm_lin_op = model._Kmm_lin_op
            K_mm_trace = model._get_Kmm_trace()
            update_params = {
                "K_mm_lin_op": K_mm_lin_op,
                "K_mm_trace": K_mm_trace,
                "n": model.n,
                "m": model.m,
                "lambd": model.lambd,
            }
        elif type == "newton":
            K_lin_op, _ = model._get_full_lin_op()
            update_params = {"K_lin_op": K_lin_op, "n": model.n}
        elif type == "nystrom":
            K_lin_op, K_trace = model._get_full_lin_op()
            update_params = {"K_lin_op": K_lin_op, "K_trace": K_trace, "n": model.n}
        elif type == "partial_cholesky":
            K_fn = model._get_kernel_fn()
            K_diag = model._get_diag()
            blk_size = precond_params.get("blk_size", None)
            update_params = {
                "K_fn": K_fn,
                "K_diag": K_diag,
                "x": model.x,
                "blk_size": blk_size,
            }
    precond = pi._get_precond(precond_params, update_params, model.lambd, model.device)

    return precond
