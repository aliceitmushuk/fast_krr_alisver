import torch
import numpy as np

from .opt_utils_sgd import (
    _get_needed_quantities_inducing,
    _get_precond_L_inducing,
    _get_stochastic_grad_diff_inducing,
    _get_full_grad_inducing,
    _apply_precond,
)


class SketchyKatyusha:
    def __init__(self, bg, bH=None, p=None, mu=None, precond_params=None):
        self.bg = bg
        self.bH = bH
        self.p = p
        self.mu = mu
        self.precond_params = precond_params

        self.theta2 = 0.5

    def run(
        self,
        x,
        b,
        x_tst,
        b_tst,
        kernel_params,
        inducing_pts,
        lambd,
        task,
        a0,
        max_iter,
        device,
        logger=None,
    ):

        x_inducing_j, K_mm, K_nm, K_tst, m, n, b_norm = _get_needed_quantities_inducing(
            x, x_tst, inducing_pts, kernel_params, b
        )

        K_nmTb = K_nm.T @ b  # Useful for computing metrics

        if logger is not None:
            logger_enabled = True

            def metric_lin_op(v):
                return K_nm.T @ (K_nm @ v) + lambd * (K_mm @ v)

        if logger_enabled:
            logger.reset_timer()

        # Set hyperparameters if not provided
        if self.bH is None:
            self.bH = int(n**0.5)

        precond, L = _get_precond_L_inducing(
            x,
            m,
            n,
            self.bH,
            x_inducing_j,
            kernel_params,
            K_mm,
            lambd,
            self.precond_params,
            device,
        )

        # Set hyperparameters if not provided
        if self.p is None:
            self.p = self.bg / n
        if self.mu is None:
            self.mu = lambd

        sigma = self.mu / L
        theta1 = min(torch.sqrt(2 / 3 * n * sigma), 0.5)
        eta = self.theta2 / ((1 + self.theta2) * theta1)

        a = a0.clone()
        y = a0.clone()
        z = a0.clone()
        g_bar = _get_full_grad_inducing(K_nm, K_mm, y, b, lambd)

        if (
            logger_enabled
        ):  # We use K_nmTb instead of b because we are using inducing points
            logger.compute_log_reset(
                metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, -1, True
            )

        for i in range(max_iter):
            w = theta1 * z + self.theta2 * y + (1 - theta1 - self.theta2) * a

            # TODO: Use a shuffling approach instead of random sampling to match PROMISE
            idx = torch.from_numpy(np.random.choice(n, self.bg, replace=False))
            g_diff = _get_stochastic_grad_diff_inducing(
                x, n, idx, x_inducing_j, kernel_params, K_mm, w, y, b, lambd
            )
            dir = _apply_precond(g_diff + g_bar, precond)

            z_new = 1 / (1 + eta * sigma) * (eta * sigma * w + z - eta / L * dir)

            # Update parameters
            a = w + theta1 * (z_new - z)

            z = z_new.clone()

            # Update snapshot
            if torch.rand(1).item() < self.p:
                y = a.clone()
                g_bar = _get_full_grad_inducing(K_nm, K_mm, y, b, lambd)

            if logger_enabled:
                logger.compute_log_reset(
                    metric_lin_op, K_tst, a, K_nmTb, b_tst, b_norm, task, i, True
                )

        return a
