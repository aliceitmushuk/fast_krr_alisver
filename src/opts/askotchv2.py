import torch

from .opt_utils import _get_leverage_scores
from .opt_utils_bcd import (
    _get_block_update,
    _get_block_properties,
)


class ASkotchV2:
    def __init__(
        self,
        model,
        block_sz,
        sampling_method="rls",
        precond_params=None,
        mu=None,
        nu=None,
        accelerated=True,
    ):
        self.model = model
        self.block_sz = block_sz
        self.precond_params = precond_params
        self.mu = mu if mu is not None else self.model.lambd
        self.nu = nu if nu is not None else self.model.n / self.block_sz
        self.accelerated = accelerated

        # TODO(pratik): check that nu > mu

        # TODO(pratik): try automatically setting eta
        # Idea: take a bunch of randomly sampled blocks (according to leverage scores),
        # and compute eta via powering
        # Then take the geometric mean of these etas to set the stepsize

        # Compute sampling probabilities
        if sampling_method == "rls":
            leverage_scores = _get_leverage_scores(
                model=self.model,
                size_final=int(self.model.n**0.5),
                lam_final=self.model.lambd,
                rls_oversample_param=5,
            )
            self.probs = leverage_scores / torch.sum(leverage_scores)
        elif sampling_method == "uniform":
            self.probs = torch.ones(self.model.n) / self.model.n

        if self.accelerated:
            self.beta = 1 - (self.mu / self.nu) ** 0.5
            self.gamma = 1 / (self.mu * self.nu) ** 0.5
            self.alpha = 1 / (1 + self.gamma * self.nu)

            self.v = self.model.w.clone()
            self.y = self.model.w.clone()

    def step(self):
        # Randomly select block_sz distinct indices
        block = torch.multinomial(self.probs, self.block_sz, replacement=False)

        # Compute block preconditioner and learning rate
        block_precond, block_eta, _ = _get_block_properties(
            self.model, self.precond_params, [block], False
        )
        block_precond = block_precond[0]
        block_eta = block_eta[0]

        # Get the update direction
        # Update direction is computed at self.y if accelerated, else at self.model.w
        eval_loc = self.y if self.accelerated else self.model.w
        dir = _get_block_update(self.model, eval_loc, block, block_precond)

        if self.accelerated:
            self.model.w = self.y.clone()
            self.model.w[block] -= block_eta * dir
            self.v = self.beta * self.v + (1 - self.beta) * self.y
            self.v[block] -= block_eta * self.gamma * dir
            self.y = self.alpha * self.v + (1 - self.alpha) * self.model.w
        else:
            self.model.w[block] -= block_eta * dir
