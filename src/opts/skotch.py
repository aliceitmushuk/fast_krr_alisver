import torch

from .opt_utils_bcd import (
    _get_blocks,
    _get_block_precond,
    _get_block_properties,
    _get_block_update,
)


class Skotch:
    def __init__(self, model, B, no_store_precond, alpha=0.5, precond_params=None):
        self.model = model
        self.B = B
        self.no_store_precond = no_store_precond
        self.alpha = alpha
        self.precond_params = precond_params

        self.blocks = _get_blocks(self.model.n, self.B)
        self.block_preconds, self.block_etas, self.block_Ls = _get_block_properties(
            self.model, self.blocks, self.precond_params, self.no_store_precond
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

        # Retrieve the block preconditioner -- recompute if necessary
        if self.no_store_precond:
            block_precond, _ = _get_block_precond(
                self.model, self.blocks[block_idx], self.precond_params
            )
        else:
            block_precond = self.block_preconds[block_idx]

        # Get the block, step size, and update direction
        block, eta, dir = _get_block_update(
            self.model,
            self.model.w,
            self.blocks[block_idx],
            block_precond,
            self.block_etas[block_idx],
        )

        # Update block
        self.model.w[block] -= eta * dir
