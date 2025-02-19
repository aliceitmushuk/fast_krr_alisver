import torch

from fast_krr.opts.optimizer import Optimizer
from fast_krr.opts._utils.minibatch_generator import MinibatchGenerator
from fast_krr.opts._utils.general import _apply_precond
from fast_krr.opts._utils.sgd import _get_precond_L, _get_minibatch


class Mimosa(Optimizer):
    def __init__(self, model, bg, bH, bH2, precond_params=None):
        super().__init__(model, precond_params)
        self.bg = bg
        self.bH = bH
        self.bH2 = bH2

        self.precond, L = _get_precond_L(
            self.model, self.precond_params, self.bH, self.bH2
        )
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
        dir = _apply_precond(g + self.model._get_grad_regularizer(), self.precond)

        # Update parameters
        self.model.w -= self.eta * dir
