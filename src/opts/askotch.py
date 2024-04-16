import torch

from .opt_utils_bcd import (
    _get_blocks,
    _get_block_properties,
    _get_block_update,
)


class ASkotch:
    def __init__(self, model, B, beta=0, precond_params=None):
        self.model = model
        self.B = B
        self.beta = beta
        self.precond_params = precond_params

    def run(
        self,
        max_iter,
        logger=None,
    ):
        blocks = _get_blocks(self.model.n, self.B)

        alpha = (1 - self.beta) / 2  # Controls acceleration

        logger_enabled = False
        if logger is not None:
            logger_enabled = True

        if logger_enabled:
            logger.reset_timer()

        block_preconds, block_etas, block_Ls = _get_block_properties(
            self.model, blocks, self.precond_params
        )

        S_alpha = sum([L**alpha for L in block_Ls])

        block_probs = torch.tensor([L**alpha / S_alpha for L in block_Ls])
        sampling_dist = torch.distributions.categorical.Categorical(block_probs)
        tau = 2 / (1 + (4 * (S_alpha**2) / self.model.lambd + 1) ** 0.5)
        gamma = 1 / (tau * S_alpha**2)

        y = self.model.w.clone()
        z = self.model.w.clone()

        if logger_enabled:
            logger.compute_log_reset(
                self.model.lin_op, self.model.K_tst, y, self.model.b, self.model.b_tst, self.model.b_norm, self.model.task, -1, False
            )

        for i in range(max_iter):
            # Randomly select a block
            block_idx = sampling_dist.sample()

            # Get the block, step size, and update direction
            block, eta, dir = _get_block_update(
                self.model,
                block_idx,
                blocks,
                block_preconds,
                block_etas
            )

            # Update y
            y = self.model.w.clone()
            y[block] -= eta * dir

            # Update z
            z = (1 / (1 + gamma * self.model.lambd)) * (z + gamma * self.model.lambd * self.model.w)
            z[block] -= (
                (1 / (1 + gamma * self.model.lambd))
                * gamma
                / (block_probs[block_idx] * (block_Ls[block_idx] ** self.beta))
                * dir
            )

            # Update w
            self.model.w = tau * z + (1 - tau) * y

            if logger_enabled:
                logger.compute_log_reset(
                    self.model.lin_op, self.model.K_tst, y, self.model.b, self.model.b_tst, self.model.b_norm, self.model.task, i, False
                )
