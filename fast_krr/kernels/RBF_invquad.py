import torch

from fast_krr.kernels.kernel import Kernel


class Rbf_invquad(Kernel):
    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for RBF kernel")

    @staticmethod
    def _get_kernel(x1_lazy, x2_lazy, kernel_params):
        Rbf._check_kernel_params(kernel_params)

        D = ((x1_lazy - x2_lazy) ** 2).sum(dim=2)
        K = 1/((1+(D*kernel_params["sigma"])**2).sqrt())

        return K

    @staticmethod
    def _get_row(x_i, x, kernel_params):
        D = ((x_i - x) ** 2).sum(dim=1)
        return 1/((1+(D*kernel_params["sigma"])**2).sqrt())

    @staticmethod
    def _get_diag(n):
        return torch.ones(n)

    @staticmethod
    def _get_trace(n):
        return Rbf_invquad._get_diag(n).sum().item()
