import torch


class Falkon:
    def __init__(self, device):
        self.device = device
        self.T = None
        self.R = None

    def update(self, K_mm_lz, n, m, lambd):
        # Instantiate K_mm_lz as dense tensor
        K_mm = K_mm_lz @ torch.eye(m).to(self.device)

        # Shift factor for numerical stability
        shift = 1e-16 * torch.trace(K_mm)

        # Get preconditioning matrices via Cholesky factorization
        T = torch.linalg.cholesky(
            K_mm + shift * torch.eye(m).to(self.device), upper=True
        )
        R = torch.linalg.cholesky(
            n / m * (T @ T.T) + lambd * torch.eye(m).to(self.device), upper=True
        )

        self.T = T
        self.R = R

    def inv_lin_op(self, v):
        # Computes T\(R\(R.T\(T.T\v)))
        v = v.reshape(v.shape[0], 1)
        v = torch.linalg.solve_triangular(self.T.T, v, upper=False)
        v = torch.linalg.solve_triangular(self.R.T, v, upper=False)
        v = torch.linalg.solve_triangular(self.R, v, upper=True)
        v = torch.linalg.solve_triangular(self.T, v, upper=True)
        return v.reshape(
            v.shape[0],
        )
