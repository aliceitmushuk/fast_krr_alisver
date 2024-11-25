import torch


class Newton:
    def __init__(self, device, rho, lambd):
        self.device = device
        self.rho = self._get_damping(rho, lambd)
        self.L = None

    def update(self, K_lin_op, n):
        K = K_lin_op(torch.eye(n, device=self.device))
        K.diagonal().add_(self.rho)  # Add self.rho to the diagonal in-place
        self.L = torch.linalg.cholesky(K)

    def _get_damping(self, rho, lambd):
        if isinstance(rho, float):
            return rho
        elif rho == "regularization":
            return lambd

    def inv_lin_op(self, v):
        v = torch.linalg.solve_triangular(self.L, v.unsqueeze(-1), upper=False)
        v = torch.linalg.solve_triangular(self.L.T, v, upper=True)
        return v.squeeze()
