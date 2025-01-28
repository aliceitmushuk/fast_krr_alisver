import torch

from .optimizer import EigenPro
from .utils.sgd import _get_minibatch


class EigenPro4(EigenPro):
    def __init__(
        self,
        model,
        bg: int,
        block_sz: int,
        r: int,
        proj_freq: int,
    ):
        super().__init__(model, bg, block_sz, r)
        self.proj_freq = proj_freq
        self._apply_precond, self.eta, self.block = self._setup()
        self.Kzb = self.K_fn(self.model.inducing_pts, self.block, get_row=False)
        self._reset()

    def _setup(self):
        eigvals, eigvecs, beta, tail_eigval, block = self._get_top_eigensys()
        diag = (1 - (tail_eigval / eigvals)) / eigvals

        def _apply_precond(v, kmat):
            return eigvecs @ (diag * (eigvecs.T @ (kmat @ v)))

        eta = self._compute_eta(tail_eigval, beta)

        return _apply_precond, eta, block

    def _reset(self):
        self.Z_tmp = torch.empty(0, device=self.model.device)
        self.alpha_tmp = torch.empty(0, device=self.model.device)
        self.alpha_b = torch.zeros(self.block_sz, device=self.model.device)
        self.h = torch.zeros_like(self.model.w, device=self.model.device)
        self.n_inner_iters = 0

    def _project(self):
        pass

    def step(self):
        idx = _get_minibatch(self.generator)
        Kmz = self.K_fn(self.model.x[idx], self.model.inducing_pts, get_row=False)
        grad = Kmz @ self.model.w - self.model.b[idx]
        if self.Z_tmp.shape[0] > 0:
            Kmz_temp = self.K_fn(self.model.x[idx], self.Z_tmp, get_row=False)
            grad += Kmz_temp @ self.alpha_tmp
        Kbm = self.K_fn(self.block, self.model.x[idx], get_row=False)
        grad += Kbm.T @ self.alpha_b

        self.Z_tmp = torch.cat([self.Z_tmp, idx])
        self.alpha_tmp = torch.cat([self.alpha_tmp, -self.eta * grad])

        self.alpha_b += self.eta * self._apply_precond(grad, Kbm)
        self.h -= self.eta * Kmz.T @ grad
        self.h += self.eta * self.Kzb @ self._apply_precond(grad, Kbm)

        self.n_inner_iters += 1
        if self.n_inner_iters == self.proj_freq:
            theta = self._project()
            self.model.w -= self.model.n / self.model.m * self.eta * theta
            self._reset()
