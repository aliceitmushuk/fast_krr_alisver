import torch

from .optimizer import Optimizer
from .utils.bcd import (
    _get_blocks,
    _get_block_precond,
    _get_block_properties,
    _get_block_update,
)


class ASkotch(Optimizer):
    def __init__(
        self,
        model,
        B,
        no_store_precond=True,
        precond_params=None,
        beta=0,
        accelerated=True,
    ):
        super().__init__(model, precond_params)
        self.B = B
        self.no_store_precond = no_store_precond
        self.beta = beta
        self.accelerated = accelerated

        self.blocks = _get_blocks(self.model.n, self.B)
        self.alpha = (1 - self.beta) / 2  # Controls acceleration
        self.block_preconds, self.block_etas, self.block_Ls = _get_block_properties(
            self.model, self.precond_params, self.blocks, self.no_store_precond
        )
        self.S_alpha = sum([L**self.alpha for L in self.block_Ls])
        self.block_probs = torch.tensor(
            [L**self.alpha / self.S_alpha for L in self.block_Ls]
        )
        self.sampling_dist = torch.distributions.categorical.Categorical(
            self.block_probs
        )

        if self.accelerated:
            self.tau = 2 / (1 + (4 * (self.S_alpha**2) / self.model.lambd + 1) ** 0.5)
            self.gamma = 1 / (self.tau * self.S_alpha**2)

            self.y = self.model.w.clone()
            self.z = self.model.w.clone()

    def step(self):
        # Randomly select a block
        block_idx = self.sampling_dist.sample()

        # Retrieve the block preconditioner -- recompute if necessary
        if self.no_store_precond:
            block_precond, _ = _get_block_precond(
                self.model, self.precond_params, self.blocks[block_idx]
            )
        else:
            block_precond = self.block_preconds[block_idx]
        block = self.blocks[block_idx]
        block_eta = self.block_etas[block_idx]

        # Get the update direction
        dir = _get_block_update(self.model, self.model.w, block, block_precond)

        if self.accelerated:
            # Update y
            self.y = self.model.w.clone()
            self.y[block] -= block_eta * dir

            # Update z
            self.z = (1 / (1 + self.gamma * self.model.lambd)) * (
                self.z + self.gamma * self.model.lambd * self.model.w
            )
            self.z[block] -= (
                (1 / (1 + self.gamma * self.model.lambd))
                * self.gamma
                / (
                    self.block_probs[block_idx]
                    * (self.block_Ls[block_idx] ** self.beta)
                )
                * dir
            )

            # Update w
            self.model.w = self.tau * self.z + (1 - self.tau) * self.y
        else:
            self.model.w[block] -= block_eta * dir
