import torch
import numpy as np

from .opt_utils_sgd import _get_needed_quantities_inducing, _get_precond_L_inducing, _get_stochastic_grad_diff_inducing, _get_full_grad_inducing, _apply_precond

class SketchySVRG():
    def __init__(self, bg, bH, update_freq, precond_params=None):
        self.bg = bg
        self.bH = bH
        self.update_freq = update_freq
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
        a_tilde = None
        g_bar = None

        if logger_enabled: # We use K_nmTb instead of b because we are using inducing points
            logger.compute_log_reset(metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, -1, True)

        for i in range(max_iter):
            if i % self.update_freq == 0:
                a_tilde = a.clone()
                g_bar = _get_full_grad_inducing(K_nm, K_mm, a_tilde, b, lambd)

            # TODO: Use a shuffling approach instead of random sampling to match PROMISE
            idx = torch.from_numpy(np.random.choice(n, self.bg, replace=False))
            g_diff = _get_stochastic_grad_diff_inducing(x, n, idx, x_inducing_j, kernel_params, K_mm, a, a_tilde, b, lambd)
            dir = _apply_precond(g_diff + g_bar, precond)

            # Update parameters
            a -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, i, True)

        return a