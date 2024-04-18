import torch

from .opt_utils_bcd import (
    _get_blocks,
    _get_block_properties,
    _get_block_update,
)


class Skotch:
    def __init__(self, model, B, alpha=0.5, precond_params=None):
        self.model = model
        self.B = B
        self.alpha = alpha
        self.precond_params = precond_params

        self.blocks = _get_blocks(self.model.n, self.B)
        self.block_preconds, self.block_etas, self.block_Ls = _get_block_properties(
            self.model, self.blocks, self.precond_params
        )
        self.S_alpha = sum([L**self.alpha for L in self.block_Ls])
        self.block_probs = torch.tensor(
            [L**self.alpha / self.S_alpha for L in self.block_Ls]
        )
        self.sampling_dist = torch.distributions.categorical.Categorical(
            self.block_probs
        )

    def step(self):
        # Randomly select a block
        block_idx = self.sampling_dist.sample()

        # Get the block, step size, and update direction
        block, eta, dir = _get_block_update(
            self.model,
            self.model.w,
            block_idx,
            self.blocks,
            self.block_preconds,
            self.block_etas,
        )

        # Update block
        self.model.w[block] -= eta * dir
