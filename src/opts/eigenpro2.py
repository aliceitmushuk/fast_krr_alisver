import torch

from .optimizer import Optimizer
from .utils.minibatch_generator import MinibatchGenerator
from .utils.bcd import _get_block


class EigenPro2(Optimizer):
    # Based on https://github.com/EigenPro/EigenPro-pytorch/tree/master
    def __init__(
        self,
        model,
        bg: int,
        block_sz: int,
        r: int,
        alpha: float,
    ):
        super().__init__(model, None)
        self.bg = bg
        self.block_sz = block_sz
        self.r = r
        self.alpha = alpha
        self.generator = MinibatchGenerator(self.model.n, self.bg)
        self.probs = torch.ones(self.model.n) / self.model.n
        self.probs_cpu = self.probs.cpu().numpy()

    def _setup(self):
        block = _get_block(self.probs, self.probs_cpu, self.block_sz)
        block_lin_op, _, _ = self.model._get_block_lin_ops(block)
        Ks = block_lin_op(torch.eye(block.shape[0], device=self.model.device))
        eigvals, eigvecs = torch.lobpcg(Ks / Ks.shape[0], self.r + 1)
        beta = Ks.diag().max()  # noqa: F841
        eigvecs = eigvecs / (Ks.shape[0] ** 0.5)
