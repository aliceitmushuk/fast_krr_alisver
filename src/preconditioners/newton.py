import torch


class Newton:
    def __init__(self, device):
        self.device = device
        self.L = None

    def update(self, K_lin_op, n):
        K = K_lin_op(torch.eye(n, device=self.device))
        self.L = torch.linalg.cholesky(K)

    def inv_lin_op(self, v):
        v = torch.linalg.solve_triangular(self.L, v.unsqueeze(-1), upper=False)
        v = torch.linalg.solve_triangular(self.L.T, v, upper=True)
        return v.squeeze()
