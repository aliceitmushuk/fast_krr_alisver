from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L,
    _apply_precond,
    _get_minibatch,
)


class SketchySVRG:
    def __init__(self, model, bg, bH, update_freq, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.update_freq = update_freq
        self.precond_params = precond_params

        self.precond, L = _get_precond_L(self.model, self.bH, self.precond_params)
        self.eta = 0.5 / L
        self.generator = MinibatchGenerator(self.model.n, self.bg)

        self.w_tilde = None
        self.g_bar = None

        self.n_iter = 0

    def step(self):
        if self.n_iter % self.update_freq == 0:
            self.w_tilde = self.model.w.clone()
            self.g_bar = self.model._get_full_grad(self.w_tilde)

        idx = _get_minibatch(self.generator)
        g_diff = self.model._get_stochastic_grad_diff(idx, self.model.w, self.w_tilde)
        dir = _apply_precond(g_diff + self.g_bar, self.precond)

        # Update parameters
        self.model.w -= self.eta * dir

        self.n_iter += 1
