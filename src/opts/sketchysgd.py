from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L_inducing,
    _apply_precond,
    _get_minibatch,
)


class SketchySGD:
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

        precond, L = _get_precond_L_inducing(self.model, self.bH, self.precond_params)

        eta = 0.5 / L

        if (
            logger_enabled
        ):  # We use K_nmTb instead of b because we are using inducing points
            logger.compute_log_reset(
                self.model.lin_op, self.model.K_tst, self.model.w, self.model.K_nmTb, self.model.b_tst, self.model.b_norm, self.model.task, -1, True
            )

        generator = MinibatchGenerator(self.model.n, self.bg)

        for i in range(max_iter):
            idx = _get_minibatch(generator)
            g = self.model._get_stochastic_grad(idx, self.model.w)
            dir = _apply_precond(g, precond)

            # Update parameters
            self.model.w -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(
                    self.model.lin_op, self.model.K_tst, self.model.w, self.model.K_nmTb, self.model.b_tst, self.model.b_norm, self.model.task, i, True
                )
