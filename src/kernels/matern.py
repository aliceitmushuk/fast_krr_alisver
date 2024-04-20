import torch

from .kernel import Kernel


class Matern(Kernel):
    def __init__(self, x1_lazy, x2_lazy, kernel_params):
        super().__init__(x1_lazy, x2_lazy, kernel_params)

    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for Matern kernel")
        if "nu" not in kernel_params:
            raise ValueError("Missing nu for Matern kernel")
        if kernel_params["nu"] not in [0.5, 1.5, 2.5]:
            raise ValueError("nu must be 0.5, 1.5, or 2.5")

    def _compute_kernel(self, x1_lazy, x2_lazy, kernel_params):
        nu = kernel_params["nu"]
        sigma = kernel_params["sigma"]

        D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2).sqrt()

        if nu == 0.5:
            K = (-D / sigma).exp()
        elif nu == 1.5:
            D_adj = torch.sqrt(torch.tensor(3.0)) * D / sigma
            K = (1 + D_adj) * (-D_adj).exp()
        else:  # nu == 2.5
            D_adj = torch.sqrt(torch.tensor(5.0)) * D / sigma
            K = (1 + D_adj + 5 * D**2 / (3 * sigma**2)) * (-D_adj).exp()

        return K

    def get_diag(self):
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError("The kernel matrix is not square")
        return torch.ones(self.K.shape[0])

    def get_trace(self):
        return self.get_diag().sum().item()