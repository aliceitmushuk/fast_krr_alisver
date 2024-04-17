import torch

from .opt_utils_pcg import (
    _get_precond,
    _get_precond_inducing,
    _init_pcg,
    _step_pcg,
)


class PCG:
    def __init__(self, model, precond_params=None):
        self.model = model
        self.precond_params = precond_params

    def run(self, max_iter, logger=None, pcg_tol=1e-6):
        logger_enabled = False
        if logger is not None:
            logger_enabled = True

        if logger_enabled:
            logger.reset_timer()

        if self.model.inducing:
            precond = _get_precond_inducing(
                self.model, self.precond_params, self.model.device
            )
            rhs = self.model.K_nmTb
        else:
            precond = _get_precond(self.model, self.precond_params, self.model.device)
            rhs = self.model.b

        r, z, p = _init_pcg(self.model.w, self.model.lin_op, rhs, precond)

        if logger_enabled:
            logger.compute_log_reset(
                self.model.lin_op,
                self.model.K_tst,
                self.model.w,
                rhs,
                self.model.b_tst,
                self.model.b_norm,
                self.model.task,
                -1,
                self.model.inducing,
            )

        for i in range(max_iter):
            self.model.w, r, z, p = _step_pcg(
                self.model.w, r, z, p, self.model.lin_op, precond
            )

            if logger_enabled:
                logger.compute_log_reset(
                    self.model.lin_op,
                    self.model.K_tst,
                    self.model.w,
                    rhs,
                    self.model.b_tst,
                    self.model.b_norm,
                    self.model.task,
                    i,
                    self.model.inducing,
                )

            if torch.norm(r) < pcg_tol:
                print(
                    f"PCG has converged with residual {torch.norm(r)} at iteration {i}"
                )
                break
