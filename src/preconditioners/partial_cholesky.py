import torch


class PartialCholesky:
    def __init__(self, device, r, rho=None):
        self.device = device
        self.r = r
        self.rho = rho

        self.U = None
        self.S = None

    def update(self, x, kernel_params, K, diag_K, tol=10**-6):
        n = x.shape[0]
        L = torch.zeros(self.r, n).to(self.device)
        diag_K = diag_K.to(self.device)

        for i in range(self.r):
            # Find pivot index corresponding to maximum diagonal entry
            idx = torch.argmax(diag_K)

            # Fetch corresponding datapoint and compute corresponding row of the kernel
            x_idx = x[idx]
            k_idx = K.get_row(x_idx, x, kernel_params)

            # Cholesky factor update
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
