import torch
import numpy as np

from .opt_utils_sgd import _get_needed_quantities_inducing, _get_precond_L_inducing, _get_table_val, _apply_precond

class SketchySAGA():
    def __init__(self, bg, bH, precond_params=None):
        self.bg = bg
        self.bH = bH
        self.precond_params = precond_params

    def run(self, x, b, x_tst, b_tst, kernel_params, inducing_pts, lambd, task,
            a0, max_iter, device, logger=None):

        x_inducing_j, K_mm, K_nm, K_tst, m, n, b_norm = _get_needed_quantities_inducing(
            x, x_tst, inducing_pts, kernel_params, b)

        K_nmTb = K_nm.T @ b  # Useful for computing metrics

        if logger is not None:
            logger_enabled = True

            def metric_lin_op(v): return K_nm.T @ (K_nm @
                                                   v) + lambd * (K_mm @ v)

        if logger_enabled:
            logger.reset_timer()

        precond, L = _get_precond_L_inducing(x, m, n, self.bH, x_inducing_j, kernel_params,
                                             K_mm, lambd, self.precond_params, device)

        eta = 0.5 / L

        a = a0.clone()
        table = torch.zeros(x.shape[0], device=device)
        u = torch.zeros(m, device=device) # Running average in SAGA

        if logger_enabled: # We use K_nmTb instead of b because we are using inducing points
                    logger.compute_log_reset(metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, -1, True)

        for i in range(max_iter):
            # TODO: Use a shuffling approach instead of random sampling to match PROMISE
            idx = torch.from_numpy(np.random.choice(n, self.bg, replace=False))

            # Compute the aux vector
            new_weights, K_nm_idx = _get_table_val(x, n, idx, x_inducing_j, kernel_params, a, b)
            aux = K_nm_idx.T @ (new_weights - table[idx])

            g = u + 1/idx.shape[0] * aux

            u += 1/n * aux

            # Update the table at the sampled indices
            table[idx] = new_weights

            # Update parameters, taking regularization into account
            dir = _apply_precond(g + lambd * (K_mm @ a), precond)

            # Update parameters
            a -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, i, True)

        return a