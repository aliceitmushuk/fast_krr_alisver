import torch

def _get_L(K_lin_op, lambd, precond_inv_sqrt_lin_op, n, device):
    v = torch.randn(n, device=device)
    v = v / torch.linalg.norm(v)

    max_eig = None

    for _ in range(10):  # TODO: Make this a parameter or check tolerance instead
        v_old = v.clone()

        v = precond_inv_sqrt_lin_op(v)
        v = K_lin_op(v) + lambd * v
        v = precond_inv_sqrt_lin_op(v)

        max_eig = torch.dot(v_old, v)

        v = v / torch.linalg.norm(v)

    return max_eig
