from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L,
    _apply_precond,
    _get_minibatch,
)


class SketchySGD:
    def __init__(self, model, bg, bH=None, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.precond_params = precond_params

        # Set hyperparameters if not provided
        if self.bH is None:
            self.bH = int(self.model.n**0.5)

        self.precond, L = _get_precond_L(self.model, self.bH, self.precond_params)
        self.eta = 0.5 / L
        self.generator = MinibatchGenerator(self.model.n, self.bg)

    def step(self):
        idx = _get_minibatch(self.generator)
        g = self.model._get_stochastic_grad(idx, self.model.w)
        dir = _apply_precond(g, self.precond)

        # Update parameters
        self.model.w -= self.eta * dir
