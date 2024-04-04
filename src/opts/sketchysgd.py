import torch
import numpy as np

from .opt_utils_sgd import _get_needed_quantities_inducing, _get_precond_L_inducing, _get_update_inducing

class SketchySGD():
    def __init__(self, bg, bH, precond_params=None):
        self.bg = bg
        self.bH = bH
        self.precond_params = precond_params

    def run(self, x, b, x_tst, b_tst, kernel_params, inducing_pts, lambd, task,
            a0, max_iter, device, logger=None):

        x_inducing_j, K_mm, K_nm, K_tst, m, n, b_norm = _get_needed_quantities_inducing(
            x, x_tst, inducing_pts, kernel_params, b)

        K_nmTb = K_nm.T @ b # Useful for computing metrics

        if logger is not None:
            logger_enabled = True
            def metric_lin_op(v): return K_nm.T @ (K_nm @ v) + lambd * (K_mm @ v)

        if logger_enabled:
            logger.reset_timer()

        precond, L = _get_precond_L_inducing(x, m, n, self.bH, x_inducing_j, kernel_params,
                                            K_mm, lambd, self.precond_params, device)

        eta = 0.5 / L

        a = a0.clone()

        if logger_enabled: # We use K_nmTb instead of b because we are using inducing points
            logger.compute_log_reset(metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, -1, True)

        for i in range(max_iter):
            # TODO: Use a shuffling approach instead of random sampling to match PROMISE
            idx = torch.from_numpy(np.random.choice(n, self.bg, replace=False))
            dir = _get_update_inducing(x, n, idx, x_inducing_j, kernel_params, K_mm, a, b, lambd, precond)

            # Update parameters
            a -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, i, True)

        return a