import torch

from .opt_utils_bcd import (
    _get_block_update,
    _get_block_properties,
)


def _get_leverage_scores(K, lambd, device):
    L = torch.linalg.cholesky(K + lambd * torch.eye(K.shape[0], device=device))

    # L^-T L^-1 K
    L_inv_K = torch.linalg.solve_triangular(L, K, upper=False)
    LT_inv_L_inv_K = torch.linalg.solve_triangular(L.T, L_inv_K, upper=True)

    leverage_scores = torch.diagonal(LT_inv_L_inv_K)
    return leverage_scores


class ASkotchV2:
    def __init__(
        self, model, block_sz, mu, nu, sampling_method="rls", precond_params=None
    ):
        self.model = model
        self.block_sz = block_sz
        self.mu = mu
        self.nu = nu
        self.precond_params = precond_params

        # TODO(pratik): check that nu > mu

        # TODO(pratik): try automatically setting eta
        # Idea: take a bunch of randomly sampled blocks (according to leverage scores),
        # and compute eta via powering
        # Then take the geometric mean of these etas to set the stepsize

        # Compute leverage scores and sampling probabilities
        if sampling_method == "rls":
            leverage_scores = _get_leverage_scores(
                self.model.K @ torch.eye(self.model.n, device=self.model.device),
                self.model.lambd,
                self.model.device,
            )
            self.probs = leverage_scores / torch.sum(leverage_scores)
        elif sampling_method == "uniform":
            self.probs = torch.ones(self.model.n) / self.model.n

        # Acceleration parameters
        self.beta = 1 - (self.mu / self.nu) ** 0.5
        self.gamma = 1 / (self.mu * self.nu) ** 0.5
        self.alpha = 1 / (1 + self.gamma * self.nu)

        self.v = self.model.w.clone()
        self.y = self.model.w.clone()

    def step(self):
        # Randomly select block_sz distinct indices
        block = torch.multinomial(self.probs, self.block_sz, replacement=False)

        # Compute block preconditioner and learning rate
        block_precond, block_eta, _ = _get_block_properties(
            self.model, [block], self.precond_params, False
        )
        block_precond = block_precond[0]
        block_eta = block_eta[0]

        # Get the update direction
        dir = _get_block_update(self.model, self.y, block, block_precond)

        self.model.w = self.y.clone()
        self.model.w[block] -= block_eta * dir
        self.v = self.beta * self.v + (1 - self.beta) * self.y
        self.v[block] -= block_eta * self.gamma * dir
        self.y = self.alpha * self.v + (1 - self.alpha) * self.model.w
