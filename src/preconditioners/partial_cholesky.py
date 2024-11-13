import torch
from typing import Callable, Tuple


class PartialCholesky:
    r"""Class for implementing a low-rank partial Cholesky factorization.
    Two types of factorizations are supported 'greedy' and 'rpc'.

    The option 'greedy' implements the partially pivoted Cholesky factorization,
    which greedily selects pivots.
    See Harbrecht et al. (2012) for details.

    The option 'rpc' implements the accelerated randomized
    pivoted Cholesky algorithm from Epperly et al. (2024).

    <u> References: </u>

    1. Harbrecht, Helmut, Michael Peters, and Reinhold Schneider.
    "On the low-rank approximation by the pivoted Cholesky decomposition."
    Applied Numerical Mathematics 62, no. 4 (2012): 428-440.
    2. Epperly, Ethan N., Joel A. Tropp, and Robert J. Webber.
    "Embrace rejection: Kernel matrix approximation by accelerated
    randomly pivoted Cholesky." arXiv preprint arXiv:2410.03969 (2024).
    """

    def __init__(self, device: str, r: int, rho: float = None, mode="rpc"):
        if mode not in ["rpc", "greedy"]:
            raise ValueError(f"PartialCholesky does not support factorization: {mode}")

        self.device = device
        self.r = r
        self.rho = rho
        self.mode = mode

        self.L = None
        self.M = None

    def update(
        self,
        K_fn: Callable,
        K_diag: torch.Tensor,
        x: torch.Tensor,
        blk_size: int = None,
        tol=10**-6,
    ):

        if self.mode == "greedy":
            self._update_greedy(K_fn, K_diag, x, tol)
        else:
            if blk_size is None:
                blk_size = (torch.ceil(torch.tensor(self.r / 10)).int()).item()

            self._update_accelerated_rpc(K_fn, K_diag, x, blk_size, stoptol=tol)

    def inv_lin_op(self, v: torch.Tensor) -> torch.Tensor:
        Lv = self.L @ v
        M_inv_Lv = torch.linalg.solve_triangular(
            self.M, torch.unsqueeze(Lv, 1), upper=False, left=True
        )
        MT_inv_M_inv_Lv = torch.linalg.solve_triangular(
            self.M.t(), M_inv_Lv, upper=True, left=True
        )
        v = (v - self.L.t() @ torch.squeeze(MT_inv_M_inv_Lv, 1)) / self.rho

        return v

    def _update_greedy(
        self,
        K_fn: Callable,
        K_diag: torch.Tensor,
        x: torch.Tensor,
        tol=10**-6,
    ):
        n = x.shape[0]
        self.L = torch.zeros(self.r, n, device=self.device)
        K_diag = K_diag.to(self.device)

        for i in range(self.r):
            # Find pivot index corresponding to maximum diagonal entry
            idx = torch.argmax(K_diag)

            # Fetch corresponding datapoint and compute corresponding row of the kernel
            k_idx = K_fn(x[idx], x, get_row=True)

            # Cholesky factor update
            self.L[i, :] = (
                k_idx - torch.matmul(self.L[:i, idx].t(), self.L[:i, :])
            ) / torch.sqrt(K_diag[idx])
            # Update diagonal
            K_diag -= self.L[i, :] ** 2
            K_diag = K_diag.clip(min=0)
            if torch.max(K_diag) <= tol:
                break

        self.M = torch.linalg.cholesky(
            self.L @ self.L.t() + self.rho * torch.eye(self.r, device=self.device)
        )

    def _update_accelerated_rpc(
        self,
        K_fn: Callable,
        K_diag: torch.Tensor,
        x: torch.Tensor,
        blk_size: int,
        stoptol=1e-13,
    ):
        r"""Constructs a low-rank preconditioner from Nystrom approximation
        constructed from the accelerated RPCholesky of Epperly et al. (2024)"""

        K_diag = K_diag.to(self.device)
        n = K_diag.shape[0]
        orig_trace = sum(K_diag)

        # row ordering
        self.L = torch.zeros((self.r, n), device=self.device)
        rows = torch.zeros((self.r, n), device=self.device)

        arr_idx = torch.zeros(self.r, dtype=int, device=self.device)

        counter = 0
        while counter < self.r:
            idx = torch.multinomial(K_diag / sum(K_diag), blk_size, replacement=True)

            Kbb = self._get_block_kernel(x, idx, K_fn)
            H = Kbb - self.L[0:counter, idx].T @ self.L[0:counter, idx]
            C, accepted = self._rejection_cholesky(H)
            num_sel = len(accepted)

            if num_sel > self.r - counter:
                num_sel = self.r - counter
                accepted = accepted[:num_sel]
                C = C[:num_sel, :num_sel]

            idx = idx[accepted]

            arr_idx[counter : counter + num_sel] = idx
            rows[counter : counter + num_sel, :] = self._get_row_kernel(x, idx, K_fn)
            self.L[counter : counter + num_sel, :] = (
                rows[counter : counter + num_sel, :]
                - self.L[0:counter, idx].T @ self.L[0:counter, :]
            )
            self.L[counter : counter + num_sel, :] = torch.linalg.solve(
                C, self.L[counter : counter + num_sel, :]
            )
            K_diag -= torch.sum(self.L[counter : counter + num_sel, :] ** 2, axis=0)
            K_diag = K_diag.clip(min=0)

            counter += num_sel

            if stoptol > 0 and sum(K_diag) <= stoptol * orig_trace:
                self.L = self.L[:counter, :]
                rows = rows[:counter, :]
                break

        self.M = torch.linalg.cholesky(
            self.L @ self.L.t() + self.rho * torch.eye(self.r, device=self.device)
        )

    def _get_block_kernel(self, x: torch.Tensor, block: torch.Tensor, K_fn: Callable):
        r"""Helper function that returns explicit block kernel
        matrix as a torch tensor of size |block| x |block|"""
        b = len(block)
        if b == 1:
            return K_fn(x[block], x[block], get_row=True)
        else:
            Kbb = K_fn(x[block], x[block], get_row=False)
            return Kbb @ torch.eye(b, device=self.device)

    def _get_row_kernel(self, x, row_indices, K_fn: Callable):
        r"""Helper function that returns explicit kernel matrix formed from a
        subset of rows of K as a torch tensor of size |block| x n"""
        b = len(row_indices)
        if b == 1:
            return K_fn(x[row_indices], x, get_row=True)
        else:
            Kbn = K_fn(x[row_indices], x, get_row=False)
            return (Kbn.T @ torch.eye(b, device=self.device)).T

    def _rejection_cholesky(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Implements rejection step in accelerated RPCholesky.
        See Epperly et al. (2024) for details."""
        b = H.shape[0]
        if H.shape[0] != H.shape[1]:
            raise RuntimeError("rejection_cholesky requires a square matrix")
        if torch.trace(H) <= 0:
            raise RuntimeError("rejection_cholesky requires a strictly positive trace")
        u = torch.diag(H)

        idx = []
        C = torch.zeros((b, b), device=self.device)
        for j in range(b):
            if torch.rand(1) * u[j] < H[j, j]:
                idx.append(j)
                C[j:, j] = H[j:, j] / torch.sqrt(H[j, j])
                H[(j + 1) :, (j + 1) :] -= torch.outer(C[(j + 1) :, j], C[(j + 1) :, j])
        idx = torch.tensor(idx)
        C = C[idx[:, None], idx]
        return C, idx
