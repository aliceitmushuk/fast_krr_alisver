import torch


class Nystrom:
    def __init__(self, device, r, rho=None, use_cpu=False):
        self.device = device
        self.r = r
        self.rho = rho
        self.use_cpu = (
            use_cpu  # Perform some preconditioner calculations on CPU if True
        )

        self.U = None
        self.S = None
        self.L = None

    def update(self, K_lin_op, K_trace, n):
        device_Phi = self.device if not self.use_cpu else "cpu"

        Phi = torch.randn((n, self.r), device=device_Phi) / (n**0.5)
        Phi = torch.linalg.qr(Phi, mode="reduced")[0]
        Phi = Phi.contiguous()  # Ensure memory is contiguous to prevent pyKeOps warning

        # Calculate shift
        shift = torch.finfo(Phi.dtype).eps * K_trace

        # Try to do computations on the specified device first
        if not self.use_cpu:
            Y_shifted = K_lin_op(Phi) + shift * Phi
            cholesky_target = torch.mm(Phi.t(), Y_shifted)
        else:  # Calculate preconditioner using CPU
            cholesky_target, Y_shifted = self.batch_calculate_cholesky_target(
                K_lin_op, Phi, shift, initial_batch_size=self.r
            )

        # Will either be on self.device or "cpu" depending on self.use_cpu
        C = torch.linalg.cholesky(cholesky_target)

        try:
            B = torch.linalg.solve_triangular(C.t(), Y_shifted, upper=True, left=False)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except RuntimeError as e:
            if "CUBLAS_STATUS_EXECUTION_FAILED" in str(
                e
            ) or "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
                torch.cuda.empty_cache()
                B = torch.linalg.solve_triangular(
                    C.t().to("cpu"), Y_shifted.to("cpu"), upper=True, left=False
                ).to(self.device)
            else:
                raise e

        del cholesky_target, Y_shifted, C, Phi
        torch.cuda.empty_cache()

        U, S, _ = torch.linalg.svd(B, full_matrices=False)
        S = torch.max(torch.square(S) - shift, torch.tensor(0.0))
        if self.use_cpu:
            U, S = U.to(self.device), S.to(self.device)

        self.U = U
        self.S = S

        # Set indices of self.S that are equal to 0 to a small value
        self.S[self.S < torch.finfo(self.S.dtype).eps] = 1e-4

    def batch_calculate_cholesky_target(self, K_lin_op, Phi, shift, initial_batch_size):
        _, r = Phi.shape
        batch_size = initial_batch_size

        cholesky_target_parts = []
        Y_shifted_parts = []

        while True:
            try:
                for i in range(0, r, batch_size):
                    end = min(i + batch_size, r)
                    # Ensures computations with kernels occur on self.device
                    Phi_batch = Phi[:, i:end].to(self.device)
                    Y_batch = K_lin_op(Phi_batch)
                    Y_shifted_batch = (Y_batch + shift * Phi_batch).to("cpu")
                    cholesky_target_batch = torch.mm(Phi.t(), Y_shifted_batch)
                    cholesky_target_parts.append(cholesky_target_batch)
                    Y_shifted_parts.append(Y_shifted_batch)
                cholesky_target = torch.cat(tuple(cholesky_target_parts), dim=1)
                Y_shifted = torch.cat(tuple(Y_shifted_parts), dim=1)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise RuntimeError(
                            "Batch size of 1 failed due to insufficient memory"
                        )
                    batch_size = max(1, batch_size // 2)
                    cholesky_target_parts = []
                    Y_shifted_parts = []
                else:
                    raise e

        return cholesky_target, Y_shifted

    def inv_lin_op(self, v):
        if torch.get_default_dtype() == torch.float64:  # Use the classic implementation
            UTv = self.U.t() @ v
            v = self.U @ (UTv / (self.S + self.rho)) + 1 / (self.rho) * (
                v - self.U @ UTv
            )
        else:  # Take a more numerically stable approach
            if self.L is None:
                self.L = torch.linalg.cholesky(
                    self.rho * torch.diag(self.S**-1) + self.U.T @ self.U
                )

            UTv = self.U.t() @ v
            L_inv_UTv = torch.linalg.solve_triangular(
                self.L, torch.unsqueeze(UTv, 1), upper=False, left=True
            )
            LT_inv_L_inv_UTv = torch.linalg.solve_triangular(
                self.L.t(), L_inv_UTv, upper=True, left=True
            )
            v = (v - self.U @ torch.squeeze(LT_inv_L_inv_UTv, 1)) / self.rho

        return v

    def inv_sqrt_lin_op(self, v):
        UTv = self.U.t() @ v
        v = self.U @ (UTv / ((self.S + self.rho) ** (0.5))) + 1 / (self.rho**0.5) * (
            v - self.U @ UTv
        )

        return v
