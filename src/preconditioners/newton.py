import torch


class Newton:
    def __init__(self, device, rho):
        self.device = device
        self.rho = rho
        self.L = None

    def update(self, K_lin_op, n):
        K = K_lin_op(torch.eye(n, device=self.device))
        K.diagonal().add_(self.rho)  # Add self.rho to the diagonal in-place
        self.L = torch.linalg.cholesky(K)

    def inv_lin_op(self, v):
        v = torch.linalg.solve_triangular(self.L, v.unsqueeze(-1), upper=False)
        v = torch.linalg.solve_triangular(self.L.T, v, upper=True)
        return v.squeeze()
