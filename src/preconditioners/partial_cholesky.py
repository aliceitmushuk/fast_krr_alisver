import torch


class PartialCholesky:
    def __init__(self, device, r, rho=None):
        self.device = device
        self.r = r
        self.rho = rho

        self.L = None
        self.M = None

    def update(self, K_row_fn, K_diag, x, tol=10**-6):
        n = x.shape[0]
        self.L = torch.zeros(self.r, n, device=self.device)
        K_diag = K_diag.to(self.device)

        for i in range(self.r):
            # Find pivot index corresponding to maximum diagonal entry
            idx = torch.argmax(K_diag)

            # Fetch corresponding datapoint and compute corresponding row of the kernel
            x_idx = x[idx]
            k_idx = K_row_fn(x_idx, x)

            # Cholesky factor update
            self.L[i, :] = (
                k_idx - torch.matmul(self.L[:i, idx].t(), self.L[:i, :])
            ) / torch.sqrt(K_diag[idx])
            # Update diagonal
            K_diag -= self.L[i, :] ** 2
            K_diag = K_diag.clip(min=0)
            if torch.max(K_diag) <= tol:
                break

        self.M = torch.linalg.cholesky(
            self.L @ self.L.t() + self.rho * torch.eye(self.r, device=self.device)
        )

    def inv_lin_op(self, v):
        Lv = self.L @ v
        M_inv_Lv = torch.linalg.solve_triangular(
            self.M, torch.unsqueeze(Lv, 1), upper=False, left=True
        )
        MT_inv_M_inv_Lv = torch.linalg.solve_triangular(
            self.M.t(), M_inv_Lv, upper=True, left=True
        )
        v = (v - self.L.t() @ torch.squeeze(MT_inv_M_inv_Lv, 1)) / self.rho

        return v
