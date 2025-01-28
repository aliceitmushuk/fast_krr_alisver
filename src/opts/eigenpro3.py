from typing import Optional

import torch

from .eigenpro import EigenPro
from .eigenpro2 import EigenPro2
from ..models import FullKRR
from .utils.sgd import _get_minibatch


class EigenPro4(EigenPro):
    def __init__(
        self,
        model,
        bg: int,
        block_sz: int,
        r: int,
        proj_inner_iters: Optional[int] = 10,
    ):
        super().__init__(model, bg, block_sz, r)
        self.proj_inner_iters = proj_inner_iters
        self._apply_precond, self.eta, self.block = self._setup()
        self.Kzb = self.K_fn(
            self.model.x[self.model.inducing_pts],
            self.model.x[self.block],
            get_row=False,
        )
        self.h = torch.zeros_like(self.model.w, device=self.model.device)

    def _setup(self):
        eigvals, eigvecs, beta, tail_eigval, block = self._get_top_eigensys()
        diag = (1 - (tail_eigval / eigvals)) / eigvals

        def _apply_precond(v, kmat):
            return eigvecs @ (diag * (eigvecs.T @ (kmat @ v)))

        eta = self._compute_eta(tail_eigval, beta)

        return _apply_precond, eta, block

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
        Kbm = self.K_fn(self.model.x[self.block], self.model.x[idx], get_row=False)
        self.h = Kmz.T @ grad - self.Kzb @ self._apply_precond(grad, Kbm)
        self.model.w -= self.model.n / self.bg * self.eta * self._project(self.h)
