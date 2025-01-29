import torch

from .optimizer import Optimizer
from .utils.bcd import _get_block
from .utils.minibatch_generator import MinibatchGenerator


class EigenProBase(Optimizer):
    def __init__(
        self,
        model,
        bg: int,
        block_sz: int,
        r: int,
    ):
        super().__init__(model, None)
        self.bg = bg
        self.block_sz = block_sz
        self.r = r
        self.generator = None
        self.probs = torch.ones(self.model.n) / self.model.n
        self.probs_cpu = self.probs.cpu().numpy()
        self.K_fn = self.model._get_kernel_fn()

    def _get_top_eigensys(self):
        block = _get_block(self.probs, self.probs_cpu, self.block_sz)
        block_lin_op, _, _ = self.model._get_block_lin_ops(block)
        Ks = block_lin_op(torch.eye(block.shape[0], device=self.model.device))

        eigvals, eigvecs = torch.lobpcg(Ks / Ks.shape[0], self.r + 1)
        eigvecs = eigvecs / (Ks.shape[0] ** 0.5)
        beta = Ks.diag().max()

        eigvals, tail_eigval = eigvals[:-1], eigvals[-1]
        eigvecs = eigvecs[:, :-1]

        return eigvals, eigvecs, beta, tail_eigval, block

    def _compute_bg_eta(self, eigval, beta):
        if self.bg is None:
            self.bg = min(int(beta / eigval + 1), self.model.n // 10)
        self.generator = MinibatchGenerator(self.model.n, self.bg)

        if self.bg < beta / eigval + 1:
            eta = self.bg / beta
        else:
            eta = 2 * self.bg / (beta + (self.bg - 1) * eigval)
        return eta / self.bg
