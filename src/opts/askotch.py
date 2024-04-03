import torch

from .opt_utils import _get_needed_quantities, _get_block_properties, _get_block_update

class ASkotch():
    def __init__(self, B, beta=0, precond_params=None):
        self.B = B
        self.beta = beta
        self.precond_params = precond_params

    def run(self, x, b, x_tst, b_tst, kernel_params, lambd, task,
             a0, max_iter, device, logger=None):

        x_j, K, K_tst, b_norm, blocks = _get_needed_quantities(
            x, x_tst, kernel_params, b, self.B)

        alpha = (1 - self.beta)/2 # Controls acceleration

        if logger is not None:
            logger_enabled = True
            def metric_lin_op(v): return K @ v + lambd * v

        if logger_enabled:
            logger.reset_timer()

        block_preconds, block_etas, block_Ls = _get_block_properties(x, kernel_params,
                                                              lambd, blocks,
                                                              self.precond_params,
                                                              device)

        S_alpha = sum([L ** alpha for L in block_Ls])

        block_probs = torch.tensor([L ** alpha / S_alpha for L in block_Ls])
        sampling_dist = torch.distributions.categorical.Categorical(block_probs)
        tau = 2 / (1 + (4 * (S_alpha ** 2) / lambd + 1) ** 0.5)
        gamma = 1 / (tau * S_alpha ** 2)

        a = a0.clone()
        y = a0.clone()
        z = a0.clone()

        if logger_enabled:
            logger.compute_log_reset(metric_lin_op, K_tst, y, b, b_tst, b_norm,
                                    task, -1, False)

        for i in range(max_iter):
            # Randomly select a block
            block_idx = sampling_dist.sample()

            # Get the block, step size, and update direction
            block, eta, dir = _get_block_update(block_idx, blocks, block_preconds,
                                                block_etas, x, x_j, kernel_params,
                                                a, b, lambd)

            # Update y
            y = a.clone()
            y[block] -= eta * dir

            # Update z
            z = (1 / (1 + gamma * lambd)) * (z + gamma * lambd * a)
            z[block] -= (1 / (1 + gamma * lambd)) * \
                gamma / (block_probs[block_idx] * (block_Ls[block_idx] ** self.beta)) * dir

            # Update x
            a = tau * z + (1 - tau) * y

            if logger_enabled:
                logger.compute_log_reset(
                    metric_lin_op, K_tst, y, b, b_tst, b_norm, task, i, False)
