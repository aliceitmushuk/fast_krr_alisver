import torch

from .kernel import Kernel


class Rbf(Kernel):
    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for RBF kernel")

    @staticmethod
    def _compute_kernel(x1_lazy, x2_lazy, kernel_params):
        Rbf._check_kernel_params(kernel_params)

        D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2)
        K = (-D / (2 * kernel_params["sigma"] ** 2)).exp()

        return K

    @staticmethod
    def _get_row(x_i, x, kernel_params):
        D = ((x_i - x) ** 2).sum(dim=1)
        return (-D / (2 * kernel_params["sigma"] ** 2)).exp()

    @staticmethod
    def _get_diag(n):
        return torch.ones(n)

    @staticmethod
    def _get_trace(n):
        return Rbf._get_diag(n).sum().item()
