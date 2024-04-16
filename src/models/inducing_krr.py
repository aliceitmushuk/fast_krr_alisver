import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import _get_kernel


class InducingKRR:
    def __init__(self, x, b, x_tst, b_tst, kernel_params, inducing_pts, lambd, task, w0, device):
        self.x = x
        self.b = b
        self.x_tst = x_tst
        self.b_tst = b_tst
        self.kernel_params = kernel_params
        self.inducing_pts = inducing_pts
        self.lambd = lambd
        self.task = task
        self.w = w0
        self.device = device

        # Get inducing points kernel
        x_inducing_i = LazyTensor(self.x[self.inducing_pts][:, None, :])
        self.x_inducing_j = LazyTensor(self.x[self.inducing_pts][None, :, :])
        self.K_mm = _get_kernel(x_inducing_i, self.x_inducing_j, self.kernel_params)

        # Get kernel between full training set and inducing points
        x_i = LazyTensor(self.x[:, None, :])
        self.K_nm = _get_kernel(x_i, self.x_inducing_j, self.kernel_params)

        # Get kernel for test set
        x_tst_i = LazyTensor(self.x_tst[:, None, :])
        self.K_tst = _get_kernel(x_tst_i, self.x_inducing_j, self.kernel_params)

        self.m = self.inducing_pts.shape[0]
        self.n = self.x.shape[0]
        self.b_norm = torch.norm(self.b)

        self.K_nmTb = self.K_nm.T @ self.b # Useful for computing metrics

    def lin_op(self, v):
        return self.K_nm.T @ (self.K_nm @ v) + self.lambd * (self.K_mm @ v)