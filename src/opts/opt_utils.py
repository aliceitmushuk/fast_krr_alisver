import torch

from .bless import _bless_size, _estimate_rls_bless


def _apply_precond(v, precond):
    if precond is not None:
        return precond.inv_lin_op(v)
    else:
        return v


def _get_L(mat_lin_op, precond, n, device):
    v = torch.randn(n, device=device)
    v = v / torch.linalg.norm(v)

    max_eig = None

    for _ in range(10):  # TODO: Make this a parameter or check tolerance instead
        v_old = v.clone()

        v = _apply_precond(v, precond)
        v = mat_lin_op(v)

        max_eig = torch.dot(v_old, v)

        v = v / torch.linalg.norm(v)
    return max_eig


def _get_leverage_scores(model, size_final, lam_final, rls_oversample_param):
    K_fn = model._get_kernel_fn()
    K_diag_fn = model._get_diag_fn()
    dict_reduced, _, _ = _bless_size(
        model.x, K_fn, K_diag_fn, size_final, rls_oversample_param
    )
    rls_approx = _estimate_rls_bless(dict_reduced, model.x, K_fn, K_diag_fn, lam_final)
    return rls_approx
