import torch

from ..kernels.kernel_inits import _get_kernels_start


class FullKRR:
    def __init__(self, x, b, x_tst, b_tst, kernel_params, lambd, task, w0, device):
        self.x = x
        self.b = b
        self.x_tst = x_tst
        self.b_tst = b_tst
        self.kernel_params = kernel_params
        self.lambd = lambd
        self.task = task
        self.w = w0
        self.device = device

        self.x_j, self.K, self.K_tst = _get_kernels_start(self.x, self.x_tst, self.kernel_params)
        self.b_norm = torch.norm(self.b)

        self.n = self.x.shape[0]

    def lin_op(self, v):
        return self.K @ v + self.lambd * v
