import torch

from .kernel import Kernel


class Rbf(Kernel):
    def __init__(self, x1_lazy, x2_lazy, kernel_params):
        super().__init__(x1_lazy, x2_lazy, kernel_params)

    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for RBF kernel")

    def _compute_kernel(self, x1_lazy, x2_lazy, kernel_params):
        Rbf._check_kernel_params(kernel_params)

        D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2)
        K = (-D / (2 * kernel_params["sigma"] ** 2)).exp()

        return K

    def get_diag(self):
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError("The kernel matrix is not square")
        return torch.ones(self.K.shape[0])

    def get_trace(self):
        return self.get_diag().sum().item()

    def get_row(self, x_i, x, kernel_params):
        D = ((x_i - x) ** 2).sum(dim=1)
        return (-D / (2 * kernel_params["sigma"] ** 2)).exp()
