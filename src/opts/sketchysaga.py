import torch

from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L,
    _apply_precond,
    _get_minibatch,
)


class SketchySAGA:
    def __init__(self, model, bg, bH, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.precond_params = precond_params

        self.precond, L = _get_precond_L(self.model, self.bH, self.precond_params)
        self.eta = 0.5 / L
        self.generator = MinibatchGenerator(self.model.n, self.bg)
        self.table = torch.zeros(self.model.n, device=self.model.device)
        self.u = torch.zeros(self.model.m, device=self.model.device)

    def step(self):
        idx = _get_minibatch(self.generator)

        # Compute the new table weights and aux vector
        new_weights, aux = self.model._get_table_aux(idx, self.model.w, self.table)

        g = self.u + 1 / idx.shape[0] * aux
        self.u += 1 / self.model.n * aux

        # Update the table at the sampled indices
        self.table[idx] = new_weights

        # Update parameters, taking regularization into account
        dir = _apply_precond(
            g + self.model._get_grad_regularizer(self.model.w), self.precond
        )

        # Update parameters
        self.model.w -= self.eta * dir
