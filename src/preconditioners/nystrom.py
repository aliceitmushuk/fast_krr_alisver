import torch


class Nystrom:
    def __init__(self, device, r, rho=None):
        self.device = device
        self.r = r
        self.rho = rho

        self.U = None
        self.S = None
        self.L = None

        # self.alpha = 0.999 if torch.get_default_dtype() == torch.float32 else 1.0 # Fudge factor to try to ensure the preconditioner is positive definite
        # self.beta = 1 - (1 - self.alpha) ** 0.5

    def update(self, K_lin_op, K_trace, n):
        # Calculate sketch
        Phi = torch.randn((n, self.r), device=self.device) / (n**0.5)
        Phi = torch.linalg.qr(Phi, mode="reduced")[0]

        Y = K_lin_op(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps * K_trace
        Y_shifted = Y + shift * Phi

        # Calculate Phi^T * K * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Phi.t(), Y_shifted)

        # Perform Cholesky decomposition
        C = torch.linalg.cholesky(choleskytarget)

        B = torch.linalg.solve_triangular(C.t(), Y_shifted, upper=True, left=False)
        U, S, _ = torch.linalg.svd(B, full_matrices=False)
        S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        self.U = U
        # self.U = torch.linalg.qr(U, mode="reduced")[0] if torch.get_default_dtype() == torch.float32 else U
        self.S = S

        # Set indices of self.S that are equal to 0 to a small value
        self.S[self.S < torch.finfo(self.S.dtype).eps] = 1e-4

        # self.M = self.U * torch.sqrt(self.S)[None, :]

        # print("self.S: ", self.S)
        # print(torch.linalg.norm(self.U.t() @ self.U - torch.eye(self.r, device=self.device), ord=2))

    def inv_lin_op(self, v):
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

        # if self.L is None:
        #     self.L = torch.linalg.cholesky(
        #         self.rho * torch.diag(self.S**-1) + self.U.T @ self.U
        #     )
        #     self.M = torch.linalg.solve_triangular(self.L, self.U.T, upper=False, left=True)
        #     # Multiply M by -i
        #     self.M = self.M * -1j

        # complex_type = torch.complex64 if torch.get_default_dtype() == torch.float32 else torch.complex128

        # v = ((v + self.M.t().conj() @ (self.M @ v.to(complex_type)))).real
        # v /= self.rho

        # if self.L is None:
        #     self.L = torch.linalg.cholesky(self.rho * torch.eye(self.r, device=self.device) + self.M.t() @ self.M)
        #     print(torch.linalg.norm(self.L @ self.L.t() - (self.rho * torch.eye(self.r, device=self.device) + self.M.t() @ self.M), ord=2))

        # MTv = self.M.t() @ v
        # L_inv_MTv = torch.linalg.solve_triangular(
        #     self.L, torch.unsqueeze(MTv, 1), upper=False, left=True
        # )
        # LT_inv_L_inv_MTv = torch.linalg.solve_triangular(
        #     self.L.t(), L_inv_MTv, upper=True, left=True
        # )
        # v = (v - self.M @ torch.squeeze(LT_inv_L_inv_MTv, 1)) / self.rho

        # UTv = self.U.t() @ v
        # v = v / self.rho - self.alpha * self.U @ (UTv / self.rho - UTv / (self.S + self.rho))

        return v

    def inv_sqrt_lin_op(self, v):
        UTv = self.U.t() @ v
        v = self.U @ (UTv / ((self.S + self.rho) ** (0.5))) + 1 / (self.rho**0.5) * (
            v - self.U @ UTv
        )
        # v = 1 / (self.rho**0.5) * v - self.U @ (self.beta / (self.rho ** 0.5)
        #                                         * UTv - (self.alpha ** 0.5) / ((self.S + self.rho) ** (0.5)) * UTv)

        return v