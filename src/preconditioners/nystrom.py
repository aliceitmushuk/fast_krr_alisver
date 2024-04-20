import torch


class Nystrom:
    def __init__(self, device, r, rho=None):
        self.device = device
        self.r = r
        self.rho = rho

        self.U = None
        self.S = None

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
        self.S = S

    def inv_lin_op(self, v):
        UTv = self.U.t() @ v
        v = self.U @ (UTv / (self.S + self.rho)) + 1 / (self.rho) * (v - self.U @ UTv)

        return v

    def inv_sqrt_lin_op(self, v):
        UTv = self.U.t() @ v
        v = self.U @ (UTv / ((self.S + self.rho) ** (0.5))) + 1 / (self.rho**0.5) * (
            v - self.U @ UTv
        )

        return v
