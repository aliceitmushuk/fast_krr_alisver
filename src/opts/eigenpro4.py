from .optimizer import Optimizer
from .utils.minibatch_generator import MinibatchGenerator

import torch

from .utils.bcd import _get_block
from .utils.sgd import _get_minibatch  # noqa: F401


class EigenPro4(Optimizer):
    def __init__(
        self,
        model,
        bg: int,
        block_sz: int,
        r: int,
        proj_freq: int,
    ):
        super().__init__(model, None)
        self.bg = bg
        self.block_sz = block_sz
        self.r = r
        self.proj_freq = proj_freq
        self.generator = MinibatchGenerator(self.model.n, self.bg)

    def _setup(self):
        block = _get_block(self.probs, self.probs_cpu, self.block_sz)
        block_lin_op, _, _ = self.model._get_block_lin_ops(block)
        Ks = block_lin_op(torch.eye(block.shape[0], device=self.model.device))

        eigvals, eigvecs = torch.lobpcg(Ks / Ks.shape[0], self.r)  # noqa: F401
