import torch

from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L_inducing,
    _apply_precond,
    _get_minibatch,
)


class SketchyKatyusha:
    def __init__(self, model, bg, bH=None, p=None, mu=None, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.p = p
        self.mu = mu
        self.precond_params = precond_params

        self.theta2 = 0.5

    def run(self, max_iter, logger=None):
        logger_enabled = False
        if logger is not None:
            logger_enabled = True

        if logger_enabled:
            logger.reset_timer()

        # Set hyperparameters if not provided
        if self.bH is None:
            self.bH = int(self.model.n**0.5)

        precond, L = _get_precond_L_inducing(
            self.model, self.bH, self.precond_params)

        # Set hyperparameters if not provided
        if self.p is None:
            self.p = self.bg / self.model.n
        if self.mu is None:
            self.mu = self.model.lambd

        sigma = self.mu / L
        theta1 = min(torch.sqrt(2 / 3 * self.model.n * sigma), 0.5)
        eta = self.theta2 / ((1 + self.theta2) * theta1)

        y = self.model.w.clone()
        z = self.model.w.clone()
        g_bar = self.model._get_full_grad(y)

        if (
            logger_enabled
        ):  # We use K_nmTb instead of b because we are using inducing points
            logger.compute_log_reset(
                self.model.lin_op, self.model.K_tst, self.model.w, self.model.K_nmTb, self.model.b_tst, self.model.b_norm, self.model.task, -1, True
            )

        generator = MinibatchGenerator(self.model.n, self.bg)

        for i in range(max_iter):
            x = theta1 * z + self.theta2 * y + (1 - theta1 - self.theta2) * self.model.w

            idx = _get_minibatch(generator)
            g_diff = self.model._get_stochastic_grad_diff(idx, x, y)
            dir = _apply_precond(g_diff + g_bar, precond)

            z_new = 1 / (1 + eta * sigma) * (eta * sigma * x + z - eta / L * dir)

            # Update parameters
            self.model.w = x + theta1 * (z_new - z)

            z = z_new.clone()

            # Update snapshot
            if torch.rand(1).item() < self.p:
                y = self.model.w.clone()
                g_bar = self.model._get_full_grad(y)

            if logger_enabled:
                logger.compute_log_reset(
                    self.model.lin_op, self.model.K_tst, self.model.w, self.model.K_nmTb, self.model.b_tst, self.model.b_norm, self.model.task, i, True
                )
