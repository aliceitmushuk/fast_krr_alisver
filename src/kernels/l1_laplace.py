import torch

from .kernel import Kernel


class L1Laplace(Kernel):
    def __init__(self, x1_lazy, x2_lazy, kernel_params):
        super().__init__(x1_lazy, x2_lazy, kernel_params)

    @staticmethod
    def _check_kernel_params(kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Missing sigma for L1 Laplace kernel")

    def _compute_kernel(self, x1_lazy, x2_lazy, kernel_params):
        L1Laplace._check_kernel_params(kernel_params)

        # Compute the L1 distance (Manhattan distance) between each pair of points
        D = (x1_lazy - x2_lazy).abs().sum(dim=2)
        # Compute the L1 Laplacian kernel
        K = (-D / kernel_params["sigma"]).exp()

        return K

    def get_diag(self):
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError("The kernel matrix is not square")
        return torch.ones(self.K.shape[0])

    def get_trace(self):
        return self.get_diag().sum().item()
    
    def get_row(self ,x, x_i, kernel_params):
        D = (torch.abs(x_i-x)).sum(dim=1)
        return (-D / kernel_params["sigma"]).exp()
