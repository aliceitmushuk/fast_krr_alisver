import torch
from scipy.sparse.linalg import LinearOperator, eigs

# def _get_L(mat_lin_op, precond_inv_sqrt_lin_op, n, device):
#     v = torch.randn(n, device=device)
#     v = v / torch.linalg.norm(v)

#     max_eig = None

#     for _ in range(10):  # TODO: Make this a parameter or check tolerance instead
#         v_old = v.clone()

#         v = precond_inv_sqrt_lin_op(v)
#         v = mat_lin_op(v)
#         v = precond_inv_sqrt_lin_op(v)

#         max_eig = torch.dot(v_old, v)

#         v = v / torch.linalg.norm(v)

#     return max_eig


def _get_L(mat_lin_op, precond_inv_lin_op, n, device):
    v = torch.randn(n, device=device)
    v = v / torch.linalg.norm(v)

    def P_inv_mat(x):
        # Put x on the device as a torch tensor
        x = torch.tensor(x, device=device, dtype=torch.get_default_dtype())

        # Apply the matrix operator
        x = mat_lin_op(x)

        # Apply the preconditioner
        x = precond_inv_lin_op(x)

        # Return the result as a numpy array
        return x.cpu().numpy()

    def mat_P_inv(x):
        # Put x on the device as a torch tensor
        x = torch.tensor(x, device=device, dtype=torch.get_default_dtype())

        # Apply the preconditioner
        x = precond_inv_lin_op(x)

        # Apply the matrix operator
        x = mat_lin_op(x)

        # Return the result as a numpy array
        return x.cpu().numpy()

    M = LinearOperator((n, n), matvec=P_inv_mat, rmatvec=mat_P_inv)
    max_eig = eigs(M, k=1, which="LR", return_eigenvectors=False)[0]

    # Return real part of max_eig as a torch tensor
    return torch.tensor(max_eig, device=device).real


def _apply_precond(v, precond):
    if precond is not None:
        return precond.inv_lin_op(v)
    else:
        return v
