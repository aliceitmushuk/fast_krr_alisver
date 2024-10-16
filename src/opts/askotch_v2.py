import torch

from .opt_utils_bcd import (
    _apply_precond,
    _get_block_precond,
    # _get_block_update,
)

class ASkotchV2:
    def __init__(self, model, block_sz, mu, nu, precond_params=None):
        self.model = model
        self.block_sz = block_sz
        self.mu = mu
        self.nu = nu
        self.precond_params = precond_params

        # Acceleration parameters
        self.beta = 1 - (self.mu / self.nu) ** 0.5
        self.gamma = 1 / (self.mu * self.nu) ** 0.5
        self.alpha = 1 / (1 + self.gamma * self.nu)

        self.v = self.model.w.clone()
        self.y = self.model.w.clone()

    def step(self):
        # Randomly select block_sz distinct indices
        block = torch.randperm(self.model.n)[:self.block_sz]

        # Compute block preconditioner
        block_precond, _ = _get_block_precond(
            self.model, block, self.precond_params
        )

        gb = self.model._get_grad_block(self.y, block) # TODO(pratik): avoid direct access to model functions
        dir = _apply_precond(gb, block_precond)

        self.model.w[block] = self.y[block] - dir
        self.v[block] = self.beta * self.v[block] + (1 - self.beta) * self.y[block] - self.gamma * dir
        self.y[block] = self.alpha * self.v[block] + (1 - self.alpha) * self.model.w[block]
