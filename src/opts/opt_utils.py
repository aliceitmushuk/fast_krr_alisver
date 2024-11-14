import torch


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
