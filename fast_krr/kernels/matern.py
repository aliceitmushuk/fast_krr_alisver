import torch

from fast_krr.kernels.kernel import Kernel


class Matern(Kernel):
    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for Matern kernel")
        if "nu" not in kernel_params:
            raise ValueError("Missing nu for Matern kernel")
        if kernel_params["nu"] not in [0.5, 1.5, 2.5]:
            raise ValueError("nu must be 0.5, 1.5, or 2.5")

    @staticmethod
    def _get_kernel(x1_lazy, x2_lazy, kernel_params):
        nu = kernel_params["nu"]
        sigma = kernel_params["sigma"]

        D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2).sqrt()

        if nu == 0.5:
            K = (-D / sigma).exp()
        elif nu == 1.5:
            D_adj = (3**0.5) * D / sigma
            K = (1 + D_adj) * (-D_adj).exp()
        else:  # nu == 2.5
            D_adj = (5**0.5) * D / sigma
            K = (1 + D_adj + 5 * D**2 / (3 * sigma**2)) * (-D_adj).exp()

        return K

    @staticmethod
    def _get_row(x_i, x, kernel_params):
        nu = kernel_params["nu"]
        sigma = kernel_params["sigma"]

        D = ((x_i - x) ** 2).sum(dim=1).sqrt()

        if nu == 0.5:
            k_i = (-D / sigma).exp()
        elif nu == 1.5:
            D_adj = (3**0.5) * D / sigma
            k_i = (1 + D_adj) * (-D_adj).exp()
        else:  # nu == 2.5
            D_adj = (5**0.5) * D / sigma
            k_i = (1 + D_adj + 5 * D**2 / (3 * sigma**2)) * (-D_adj).exp()

        return k_i

    @staticmethod
    def _get_diag(n):
        return torch.ones(n)

    @staticmethod
    def _get_trace(n):
        return Matern._get_diag(n).sum().item()
