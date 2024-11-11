import torch
from ..kernels.kernel_inits import _get_kernel
from pykeops.torch import LazyTensor
from time import time
from typing import Tuple

class PartialCholesky:
    r"""Class for implementing a low-rank partial Cholesky factorization. 
    Two types of factorizations are supported 'greedy' and 'rpc'.
    
    The option 'greedy' implements the partially pivoted Cholesky factorization, which greedily selects pivots. 
    See Harbrecht et al. (2012) for details.
    
    The option 'rpc' implements the accelerated randomized pivoted Cholesky algorithm from Epperly et al. (2024).

    <u> References: </u>
    
    1. Harbrecht, Helmut, Michael Peters, and Reinhold Schneider. "On the low-rank approximation by the pivoted Cholesky decomposition." Applied Numerical Mathematics 62, no. 4 (2012): 428-440.
    2. Epperly, Ethan N., Joel A. Tropp, and Robert J. Webber. "Embrace rejection: Kernel matrix approximation by accelerated randomly pivoted Cholesky." arXiv preprint arXiv:2410.03969 (2024).    
    """
    def __init__(self, device: str, r: int, rho: float=None, mode = 'rpc'):
        if mode not in ['rpc', 'greedy']:
           raise ValueError(f"PartialCholesky does not support factorization: {mode}") 
        
        self.device = device
        self.r = r
        self.rho = rho
        self.mode = mode

        self.L = None
        self.M = None
    
    def update(self, x: torch.Tensor, kernel_params: dict, K: LazyTensor, diag_K: torch.Tensor, tol=10**-6):
        if self.mode == 'greedy':
           self._update_greedy(x, kernel_params, K, diag_K, tol)
        else:
           self._update_accelerated_rpc(x, kernel_params, diag_K, stoptol=tol)  
    
    def inv_lin_op(self, v: torch.Tensor)-> torch.Tensor:
        Lv = self.L @ v
        M_inv_Lv = torch.linalg.solve_triangular(
            self.M, torch.unsqueeze(Lv, 1), upper=False, left=True
        )
        MT_inv_M_inv_Lv = torch.linalg.solve_triangular(
            self.M.t(), M_inv_Lv, upper=True, left=True
        )
        v = (v - self.L.t() @ torch.squeeze(MT_inv_M_inv_Lv, 1)) / self.rho

        return v
    
    def _update_greedy(self, x: torch.Tensor, kernel_params: dict, K: LazyTensor, diag_K: torch.Tensor, tol=10**-6):
        n = x.shape[0]
        self.L = torch.zeros(self.r, n, device=self.device)
        diag_K = diag_K.to(self.device)

        for i in range(self.r):
            # Find pivot index corresponding to maximum diagonal entry
            idx = torch.argmax(diag_K)

            # Fetch corresponding datapoint and compute corresponding row of the kernel
            x_idx = x[idx]
            k_idx = K.get_row(x_idx, x, kernel_params)

            # Cholesky factor update
            self.L[i, :] = (
                k_idx - torch.matmul(self.L[:i, idx].t(), self.L[:i, :])
            ) / torch.sqrt(diag_K[idx])
            # Update diagonal
            diag_K -= self.L[i, :] ** 2
            diag_K = diag_K.clip(min=0)
            if torch.max(diag_K) <= tol:
                break

        self.M = torch.linalg.cholesky(
            self.L @ self.L.t() + self.rho * torch.eye(self.r, device=self.device)
        )
    
    def _update_accelerated_rpc(self, x: torch.Tensor, kernel_params: dict, diag_K: torch.Tensor, b = "auto", stoptol = 1e-13, verbose=False):
        r"""Constructs a low-rank preconditioner from Nystrom approximation constructed from the accelerated RPCholesky of Epperly et al. (2024)"""
        
        diag_K = diag_K.to(self.device)
        n = diag_K.shape[0]
        orig_trace = sum(diag_K)
    
        if b == "auto":
            b = int(torch.ceil(torch.tensor(self.r / 10)))
            auto_b = True
        else:
            auto_b = False
    
        # row ordering
        self.L = torch.zeros((self.r,n),device=self.device)
        rows = torch.zeros((self.r,n), device=self.device)
    
        arr_idx = torch.zeros(self.r, dtype=int, device=self.device)
    
        counter = 0
        while counter < self.r:
            idx = torch.multinomial(diag_K / sum(diag_K), b, replacement=True)

            if auto_b:
                start = time()
            
            Kbb = self._get_block_kernel(x, idx, kernel_params)
            H = Kbb - self.L[0:counter,idx].T @ self.L[0:counter,idx]
            C, accepted = self._rejection_cholesky(H)
            num_sel = len(accepted)

            if num_sel > self.r - counter:
                num_sel = self.r - counter
                accepted = accepted[:num_sel]
                C = C[:num_sel,:num_sel]
        
            idx = idx[accepted]

            if auto_b:
                rejection_time = time() - start
                start = time()

            arr_idx[counter:counter+num_sel] = idx
            rows[counter:counter+num_sel,:] = self._get_row_kernel(x, idx, kernel_params)
            self.L[counter:counter+num_sel,:] = rows[counter:counter+num_sel,:] - self.L[0:counter,idx].T @ self.L[0:counter,:]
            self.L[counter:counter+num_sel,:] = torch.linalg.solve(C, self.L[counter:counter+num_sel,:])
            diag_K -= torch.sum(self.L[counter:counter+num_sel,:]**2, axis=0)
            diag_K = diag_K.clip(min = 0)

            if auto_b:
                process_time = time() - start

                # Assuming rejection_time ~ A b^2 and process_time ~ C b
                # then obtaining rejection_time = process_time / 4 entails
                # b = C / 4A = (process_time / b) / 4 (rejection_time / b^2)
                #   = b * process_time / (4 * rejection_time)
                target = int(torch.ceil(torch.tensor(b * process_time / (4 * rejection_time))))
                b = max([min([target, int(torch.ceil(torch.tensor(1.5*b))), int(torch.ceil(torch.tensor(self.r/3)))]),
                     int(torch.ceil(torch.tensor(b/3))), 10])

            counter += num_sel

            if stoptol > 0 and sum(diag_K) <= stoptol * orig_trace:
                self.L = self.L[:counter,:]
                rows = rows[:counter,:]
                break
            
            if verbose:
                print("Accepted {} / {}".format(num_sel, b))
        
        self.M = torch.linalg.cholesky(
            self.L @ self.L.t() + self.rho * torch.eye(self.r, device=self.device)
        )
    
    def _get_block_kernel(self, x: torch.Tensor, block: torch.Tensor, kernel_params: dict):
        r"""Helper function that returns explicit block kernel matrix as a torch tensor of size |block| x |block|"""
        xb_i = LazyTensor(x[block][:, None, :])
        xb_j = LazyTensor(x[block][None, :, :])
        Kbb = _get_kernel(xb_i, xb_j, kernel_params)
        b = len(block)
        return Kbb@torch.eye(b, device= self.device)
    
    def _get_row_kernel(self, x, row_indices, kernel_params):
        r"""Helper function that returns explicit kernel matrix formed from a subset of rows of K as a torch tensor of size |block| x n"""
        xb_i = LazyTensor(x[row_indices][:, None, :])
        xb_j = LazyTensor(x[None, :, :])
        Kbn = _get_kernel(xb_i, xb_j, kernel_params)
        b = len(row_indices)
        return (Kbn.T@torch.eye(b, device= self.device)).T

    def _rejection_cholesky(self, H: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        r"""Implements rejection step in accelerated RPCholesky. See Epperly et al. (2024) for details."""
        b = H.shape[0]
        if H.shape[0] != H.shape[1]:
            raise RuntimeError("rejection_cholesky requires a square matrix")
        if torch.trace(H) <= 0:
            raise RuntimeError("rejection_cholesky requires a strictly positive trace")
        u = torch.diag(H)

        idx = []
        C = torch.zeros((b,b), device=self.device)
        for j in range(b):
            if torch.rand(1) * u[j] < H[j,j]:
                idx.append(j)
                C[j:,j] = H[j:,j] / torch.sqrt(H[j,j])
                H[(j+1):,(j+1):] -= torch.outer(C[(j+1):,j], C[(j+1):,j])
        idx = torch.tensor(idx)
        C = C[idx[:, None], idx]
        return C, idx
