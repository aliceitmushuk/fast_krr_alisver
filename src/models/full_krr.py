import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import _get_kernel, _get_kernels_start


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
    
    def _get_block_grad(self, w, block):
        xb_i = LazyTensor(self.x[block][:, None, :])
        Kbn = _get_kernel(xb_i, self.x_j, self.kernel_params)

        return Kbn @ w + self.lambd * self.w[block] - self.b[block]

    def _get_full_lin_op(self):
        def K_lin_op(v):
            return self.K @ v

        return K_lin_op
    
    def _get_block_lin_ops(self, block):
        xb_i = LazyTensor(self.x[block][:, None, :])
        xb_j = LazyTensor(self.x[block][None, :, :])
        Kb = _get_kernel(xb_i, xb_j, self.kernel_params)

        def Kb_lin_op(v):
            return Kb @ v

        def Kb_lin_op_reg(v):
            return Kb @ v + self.lambd * v
        
        return Kb_lin_op, Kb_lin_op_reg
