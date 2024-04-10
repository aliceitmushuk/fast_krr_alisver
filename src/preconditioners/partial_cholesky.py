import torch
from pykeops.torch import LazyTensor
from kernels import _get_rbf_kernel, _get_l1_laplace_kernel, _get_matern_kernel


class Pivoted_Cholesky:
    def __init__(self, device, r, rho=None):
        self.device = device
        self.r = r
        self.rho = rho

        self.U = None
        self.S = None

    def update(self, x, kernel_params, tol=10**-6):
        n = x.shape[0]
        L = torch.zeros(self.r, n).to(self.device)
        x_f = LazyTensor(x[:, None, :])

        # Get diagonal and function for getting kernel row corresponding to the specified kernel
        if kernel_params["type"] == "rbf":
            diag_K = torch.ones(n).to(self.device)

            def get_row(x):
                return _get_rbf_kernel(x_f, x, kernel_params["sigma"])

        elif kernel_params["type"] == "l1_laplace":
            diag_K = torch.ones(n).to(self.device)

            def get_row(x):
                return _get_l1_laplace_kernel(x_f, x, kernel_params["sigma"])

        elif kernel_params["type"] == "matern":
            diag_K = torch.ones(n).to(self.device)

            def get_row(x):
                return _get_matern_kernel(
                    x_f, x, kernel_params["sigma"], kernel_params["nu"]
                )

        for i in range(self.r):
            # Find pivot index corresponding to maximum diagonal entry
            idx = torch.argmax(diag_K)
            # Instantiate lazy tensor for datapoint corresponding to this index
            x_idx = LazyTensor(x[idx])
            # Lazy tensor for kernelized datapoint
            k_idx_lz = get_row(x_idx)
            # Instantiate kernel row as a dense tensor
            k_idx = k_idx_lz @ torch.ones(1).to(self.device)
            # Cholesky update
            L[i, :] = (k_idx - torch.matmul(L[:i, idx].t(), L[:i, :])) / torch.sqrt(
                diag_K[idx]
            )
            # Update diagonal
            diag_K -= L[i, :] ** 2
            diag_K = diag_K.clip(min=0)
            if torch.max(diag_K) <= tol:
                break
        # Return approximation in terms of approximate eigendecomposition
        U, S, _ = torch.linalg.svd(L.T, full_matrices=False)
        self.U = U
        self.S = S**2

    def inv_lin_op(self, v):
        UTv = self.U.t() @ v
        v = self.U @ (UTv / (self.S + self.rho)) + 1 / (self.rho) * (v - self.U @ UTv)

        return v
