import torch

from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L,
    _apply_precond,
    _get_minibatch,
)


class SketchySAGA:
    def __init__(self, model, bg, bH=None, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.precond_params = precond_params

    def run(self, max_iter, logger=None):
        logger_enabled = False
        if logger is not None:
            logger_enabled = True

        if logger_enabled:
            logger.reset_timer()

        # Set hyperparameters if not provided
        if self.bH is None:
            self.bH = int(self.model.n**0.5)

        precond, L = _get_precond_L(self.model, self.bH, self.precond_params)

        eta = 0.5 / L

        table = torch.zeros(self.model.n, device=self.model.device)
        u = torch.zeros(
            self.model.m, device=self.model.device
        )  # Running average in SAGA

        if logger_enabled:
            logger.compute_log_reset(-1, self.model.compute_metrics, self.model.w)

        generator = MinibatchGenerator(self.model.n, self.bg)

        for i in range(max_iter):
            idx = _get_minibatch(generator)

            # Compute the new table weights and aux vector
            new_weights, aux = self.model._get_table_aux(idx, self.model.w, table)

            g = u + 1 / idx.shape[0] * aux
            u += 1 / self.model.n * aux

            # Update the table at the sampled indices
            table[idx] = new_weights

            # Update parameters, taking regularization into account
            dir = _apply_precond(
                g + self.model.lambd * (self.model.K_mm @ self.model.w), precond
            )

            # Update parameters
            self.model.w -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(i, self.model.compute_metrics, self.model.w)
