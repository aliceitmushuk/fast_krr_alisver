import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import (
    _get_kernel,
    _get_kernels_start,
    _get_row,
    _get_trace,
    _get_diag,
)


class FullKRR:
    def __init__(
        self, x, b, x_tst, b_tst, kernel_params, Ktr_needed, lambd, task, w0, device
    ):
        self.x = x
        self.b = b
        self.x_tst = x_tst
        self.b_tst = b_tst
        self.kernel_params = kernel_params
        self.lambd = lambd
        self.task = task
        self.w = w0
        self.device = device

        self.x_j, self.K, self.K_tst = _get_kernels_start(
            self.x, self.x_tst, self.kernel_params, Ktr_needed
        )
        self.b_norm = torch.norm(self.b)

        self.n = self.x.shape[0]
        self.n_tst = self.x_tst.shape[0]

        self.inducing = False

        self.test_metric_name = (
            "test_acc" if self.task == "classification" else "test_mse"
        )

    def lin_op(self, v):
        return self.K @ v + self.lambd * v

    def compute_metrics(self, v, log_test_only):
        metrics_dict = {}
        if not log_test_only:
            v_lin_op = self.lin_op(v)
            residual = v_lin_op - self.b
            rel_residual = torch.norm(residual) / self.b_norm
            loss = 1 / 2 * torch.dot(v, v_lin_op) - torch.dot(self.b, v)

            metrics_dict["rel_residual"] = rel_residual
            metrics_dict["train_loss"] = loss

        pred = self.K_tst @ v
        if self.task == "classification":
            test_metric = torch.sum(torch.sign(pred) == self.b_tst) / self.n_tst
            metrics_dict[self.test_metric_name] = test_metric
        else:
            test_metric = 1 / 2 * torch.norm(pred - self.b_tst) ** 2 / self.n_tst
            smape = (
                torch.sum(
                    (pred - self.b_tst).abs() / ((pred.abs() + self.b_tst.abs()) / 2)
                )
                / self.n_tst
            )
            metrics_dict[self.test_metric_name] = test_metric
            metrics_dict["test_rmse"] = test_metric**0.5
            metrics_dict["smape"] = smape

        return metrics_dict

    def _get_block_grad(self, w, block):
        xb_i = LazyTensor(self.x[block][:, None, :])
        Kbn = _get_kernel(xb_i, self.x_j, self.kernel_params)

        return Kbn @ w + self.lambd * self.w[block] - self.b[block]

    def _get_full_lin_op(self):
        def K_lin_op(v):
            return self.K @ v

        K_trace = _get_trace(self.K.shape[0], self.kernel_params)

        return K_lin_op, K_trace

    def _get_block_lin_ops(self, block):
        xb_i = LazyTensor(self.x[block][:, None, :])
        xb_j = LazyTensor(self.x[block][None, :, :])
        Kb = _get_kernel(xb_i, xb_j, self.kernel_params)

        def Kb_lin_op(v):
            return Kb @ v

        def Kb_lin_op_reg(v):
            return Kb @ v + self.lambd * v

        Kb_trace = _get_trace(Kb.shape[0], self.kernel_params)

        return Kb_lin_op, Kb_lin_op_reg, Kb_trace

    def _get_diag(self):
        return _get_diag(self.n, self.kernel_params)

    def _get_kernel_fn(self):
        def K_fn(x_i, x_j, get_row):
            if get_row:
                return _get_row(x_i, x_j, self.kernel_params)  # Tensor
            else:
                x_i_lz = LazyTensor(x_i[:, None, :])
                x_j_lz = LazyTensor(x_j[None, :, :])
                return _get_kernel(x_i_lz, x_j_lz, self.kernel_params)  # LazyTensor

        return K_fn
