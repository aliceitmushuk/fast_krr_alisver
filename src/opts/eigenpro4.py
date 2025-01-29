from typing import Optional

import torch

from .eigenpro_base import EigenProBase
from .eigenpro2 import EigenPro2
from ..models import FullKRR
from .utils.sgd import _get_minibatch


class EigenPro4(EigenProBase):
    # Based on https://github.com/EigenPro/EigenPro
    def __init__(
        self,
        model,
        block_sz: int,
        r: int,
        proj_freq: int,
        bg: Optional[int] = None,
        proj_inner_iters: Optional[int] = 10,
    ):
        super().__init__(model, bg, block_sz, r)
        self.proj_freq = proj_freq
        self.proj_inner_iters = proj_inner_iters
        self._apply_precond, self.eta, self.block = self._setup()
        self.Kzb = self.K_fn(
            self.model.x[self.model.inducing_pts],
            self.model.x[self.block],
            get_row=False,
        )
        self._reset()

    def _setup(self):
        eigvals, eigvecs, beta, tail_eigval, block = self._get_top_eigensys()
        diag = (1 - (tail_eigval / eigvals)) / eigvals

        def _apply_precond(v, kmat):
            return eigvecs @ (diag * (eigvecs.T @ (kmat @ v)))

        eta = self._compute_bg_eta(tail_eigval, beta)

        return _apply_precond, eta, block

    def _reset(self):
        self.Z_tmp = torch.empty(0, device=self.model.device, dtype=torch.int64)
        self.alpha_tmp = torch.empty(0, device=self.model.device)
        self.alpha_b = torch.zeros(self.block_sz, device=self.model.device)
        self.h = torch.zeros_like(self.model.w, device=self.model.device)
        self.n_inner_iters = 0

    def _project(self, v):
        proj_model = FullKRR(
            x=self.model.x[self.model.inducing_pts],
            b=v,
            x_tst=None,
            b_tst=None,
            kernel_params=self.model.kernel_params,
            Ktr_needed=True,
            lambd=0.0,
            task="regression",
            w0=torch.zeros_like(self.model.w, device=self.model.device),
            device=self.model.device,
        )
        eigenpro2_opt = EigenPro2(
            proj_model,
            bg=proj_model.n // self.proj_inner_iters,
            block_sz=min(proj_model.n, 12000),
            r=100,
        )
        for _ in range(self.proj_inner_iters):
            eigenpro2_opt.step()
        return eigenpro2_opt.model.w

    def step(self):
        idx = _get_minibatch(self.generator)
        Kmz = self.K_fn(
            self.model.x[idx], self.model.x[self.model.inducing_pts], get_row=False
        )
        grad = Kmz @ self.model.w - self.model.b[idx]
        if self.Z_tmp.shape[0] > 0:
            Kmz_temp = self.K_fn(
                self.model.x[idx], self.model.x[self.Z_tmp], get_row=False
            )
            grad += Kmz_temp @ self.alpha_tmp
        Kbm = self.K_fn(self.model.x[self.block], self.model.x[idx], get_row=False)
        grad += Kbm.T @ self.alpha_b

        idx = idx.to(self.model.device)  # to prevent device mismatch
        self.Z_tmp = torch.cat([self.Z_tmp, idx])
        self.alpha_tmp = torch.cat([self.alpha_tmp, -self.eta * grad])

        self.alpha_b += self.eta * self._apply_precond(grad, Kbm)
        self.h -= self.eta * Kmz.T @ grad
        self.h += self.eta * self.Kzb @ self._apply_precond(grad, Kbm)

        self.n_inner_iters += 1
        if self.n_inner_iters == self.proj_freq:
            theta = self._project(self.h)
            self.model.w -= self.eta * theta
            self._reset()
