import torch

from .opt_utils_bcd import (
    _get_blocks,
    _get_block_properties,
    _get_block_update,
)


class Skotch:
    def __init__(self, model, B, alpha=0.5, precond_params=None):
        self.model = model
        self.B = B
        self.alpha = alpha
        self.precond_params = precond_params

    def run(self, max_iter, logger=None):
        blocks = _get_blocks(self.model.n, self.B)

        logger_enabled = False
        if logger is not None:
            logger_enabled = True

        if logger_enabled:
            logger.reset_timer()

        block_preconds, block_etas, block_Ls = _get_block_properties(
            self.model, blocks, self.precond_params
        )

        S_alpha = sum([L**self.alpha for L in block_Ls])

        block_probs = torch.tensor([L**self.alpha / S_alpha for L in block_Ls])
        sampling_dist = torch.distributions.categorical.Categorical(block_probs)

        if logger_enabled:
            logger.compute_log_reset(-1, self.model.compute_metrics, self.model.w)

        for i in range(max_iter):
            # Randomly select a block
            block_idx = sampling_dist.sample()

            # Get the block, step size, and update direction
            block, eta, dir = _get_block_update(
                self.model, self.model.w, block_idx, blocks, block_preconds, block_etas
            )

            # Update block
            self.model.w[block] -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(i, self.model.compute_metrics, self.model.w)
