import torch

from .minibatch_generator import MinibatchGenerator
from .opt_utils import _apply_precond
from .opt_utils_sgd import _get_precond_L, _get_minibatch


class SketchyKatyusha:
    def __init__(self, model, bg, bH, bH2, p, mu, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.bH2 = bH2
        self.p = p
        self.mu = mu
        self.precond_params = precond_params

        self.theta2 = 0.5

        self.precond, self.L = _get_precond_L(
            self.model, self.bH, self.bH2, self.precond_params
        )
        self.eta = 0.5 / self.L
        self.generator = MinibatchGenerator(self.model.n, self.bg)
        self.sigma = self.mu / self.L
        self.theta1 = min(torch.sqrt(2 / 3 * self.model.n * self.sigma), 0.5)
        self.eta = self.theta2 / ((1 + self.theta2) * self.theta1)

        self.y = self.model.w.clone()
        self.z = self.model.w.clone()
        self.g_bar = self.model._get_full_grad(self.y)

    def step(self):
        x = (
            self.theta1 * self.z
            + self.theta2 * self.y
            + (1 - self.theta1 - self.theta2) * self.model.w
        )

        idx = _get_minibatch(self.generator)
        g_diff = self.model._get_stochastic_grad_diff(idx, x, self.y)
        dir = _apply_precond(g_diff + self.g_bar, self.precond)

        z_new = (
            1
            / (1 + self.eta * self.sigma)
            * (self.eta * self.sigma * x + self.z - self.eta / self.L * dir)
        )

        # Update parameters
        self.model.w = x + self.theta1 * (z_new - self.z)

        self.z = z_new.clone()

        # Update snapshot
        if torch.rand(1).item() < self.p:
            self.y = self.model.w.clone()
            self.g_bar = self.model._get_full_grad(self.y)
