import torch

from fast_krr.kernels.kernel import Kernel


class L1Laplace(Kernel):
    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for L1 Laplace kernel")

    @staticmethod
    def _get_kernel(x1_lazy, x2_lazy, kernel_params):
        L1Laplace._check_kernel_params(kernel_params)

        # Compute the L1 distance (Manhattan distance) between each pair of points
        D = (x1_lazy - x2_lazy).abs().sum(dim=2)
        # Compute the L1 Laplacian kernel
        K = (-D / kernel_params["sigma"]).exp()

        return K

    @staticmethod
    def _get_row(x_i, x, kernel_params):
        D = (torch.abs(x_i - x)).sum(dim=1)
        return (-D / kernel_params["sigma"]).exp()

    @staticmethod
    def _get_diag(n):
        return torch.ones(n)

    @staticmethod
    def _get_trace(n):
        return L1Laplace._get_diag(n).sum().item()
