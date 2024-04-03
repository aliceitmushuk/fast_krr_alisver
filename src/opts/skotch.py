import torch

from .opt_utils import _get_needed_quantities, _get_block_properties, _get_block_update

class Skotch():
    def __init__(self, B, precond_params=None):
        self.B = B
        self.precond_params = precond_params

    def run(self, x, b, x_tst, b_tst, kernel_params, lambd, task,
             a0, max_iter, device, logger=None):

        x_j, K, K_tst, b_norm, blocks = _get_needed_quantities(
            x, x_tst, kernel_params, b, self.B)

        if logger is not None:
            logger_enabled = True
            def metric_lin_op(v): return K @ v + lambd * v

        if logger_enabled:
            logger.reset_timer()

        block_preconds, block_etas, _ = _get_block_properties(x, kernel_params,
                                                             lambd, blocks,
                                                             self.precond_params,
                                                             device)

        a = a0.clone()

        if logger_enabled:
            logger.compute_log_reset(metric_lin_op, K_tst, a, b, b_tst, b_norm, task, -1, False)

        for i in range(max_iter):
            # Randomly select a block
            block_idx = torch.randint(self.B, (1,))

            # Get the block, step size, and update direction
            block, eta, dir = _get_block_update(block_idx, blocks, block_preconds,
                                                block_etas, x, x_j, kernel_params,
                                                a, b, lambd)

            # Update block
            a[block] -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(metric_lin_op, K_tst, a, b, b_tst, b_norm, task, i, False)

        return a
