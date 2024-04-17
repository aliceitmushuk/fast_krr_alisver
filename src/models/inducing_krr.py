import numpy as np
import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import _get_kernel


class InducingKRR:
    def __init__(
        self, x, b, x_tst, b_tst, kernel_params, inducing_pts, lambd, task, w0, device
    ):
        self.x = x
        self.b = b
        self.x_tst = x_tst
        self.b_tst = b_tst
        self.kernel_params = kernel_params
        self.inducing_pts = inducing_pts
        self.lambd = lambd
        self.task = task
        self.w = w0
        self.device = device

        # Get inducing points kernel
        x_inducing_i = LazyTensor(self.x[self.inducing_pts][:, None, :])
        self.x_inducing_j = LazyTensor(self.x[self.inducing_pts][None, :, :])
        self.K_mm = _get_kernel(x_inducing_i, self.x_inducing_j, self.kernel_params)

        # Get kernel between full training set and inducing points
        x_i = LazyTensor(self.x[:, None, :])
        self.K_nm = _get_kernel(x_i, self.x_inducing_j, self.kernel_params)

        # Get kernel for test set
        x_tst_i = LazyTensor(self.x_tst[:, None, :])
        self.K_tst = _get_kernel(x_tst_i, self.x_inducing_j, self.kernel_params)

        self.m = self.inducing_pts.shape[0]
        self.n = self.x.shape[0]
        self.n_tst = self.x_tst.shape[0]
        self.b_norm = torch.norm(self.b)

        self.K_nmTb = self.K_nm.T @ self.b  # Useful for computing metrics
        self.K_nmTb_norm = torch.norm(self.K_nmTb)

        self.inducing = True

        self.test_metric_name = (
            "test_acc" if self.task == "classification" else "test_mse"
        )

    def lin_op(self, v):
        return self.K_nm.T @ (self.K_nm @ v) + self.lambd * (self.K_mm @ v)

    def compute_metrics(self, v):
        residual = self.lin_op(v) - self.K_nmTb
        rel_residual = torch.norm(residual) / self.K_nmTb_norm
        loss = 1 / 2 * (torch.dot(v, residual - self.K_nmTb) + self.b_norm**2)

        metrics_dict = {"rel_residual": rel_residual, "train_loss": loss}

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
            metrics_dict["smape"] = smape

        return metrics_dict

    def _get_stochastic_grad(self, idx, w):
        x_idx_i = LazyTensor(self.x[idx][:, None, :])
        K_nm_idx = _get_kernel(x_idx_i, self.x_inducing_j, self.kernel_params)
        g = self.n / idx.shape[0] * (
            K_nm_idx.T @ (K_nm_idx @ w - self.b[idx])
        ) + self.lambd * (self.K_mm @ w)

        return g

    def _get_stochastic_grad_diff(self, idx, w1, w2):
        x_idx_i = LazyTensor(self.x[idx][:, None, :])
        K_nm_idx = _get_kernel(x_idx_i, self.x_inducing_j, self.kernel_params)
        w_diff = w1 - w2
        g_diff = self.n / idx.shape[0] * (
            K_nm_idx.T @ (K_nm_idx @ w_diff)
        ) + self.lambd * (self.K_mm @ w_diff)

        return g_diff

    def _get_full_grad(self, w):
        return self.K_nm.T @ (self.K_nm @ w - self.b) + self.lambd * (self.K_mm @ w)

    def _get_table_aux(self, idx, w, table):
        x_idx_i = LazyTensor(self.x[idx][:, None, :])
        K_nm_idx = _get_kernel(x_idx_i, self.x_inducing_j, self.kernel_params)
        new_weights = self.n * (K_nm_idx @ w - self.b[idx])
        aux = K_nm_idx.T @ (new_weights - table[idx])
        return new_weights, aux

    def _get_subsampled_lin_ops(self, bH):
        hess_pts = torch.from_numpy(np.random.choice(self.n, bH, replace=False))
        x_hess_i = LazyTensor(self.x[hess_pts][:, None, :])
        K_sm = _get_kernel(x_hess_i, self.x_inducing_j, self.kernel_params)

        hess_pts_lr = torch.from_numpy(np.random.choice(self.n, bH, replace=False))
        x_hess_lr_i = LazyTensor(self.x[hess_pts_lr][:, None, :])
        K_sm_lr = _get_kernel(x_hess_lr_i, self.x_inducing_j, self.kernel_params)

        adj_factor = self.n / bH

        def K_inducing_sub_lin_op(v):
            return adj_factor * K_sm.T @ (K_sm @ v)

        def K_inducing_sub_Kmm_lin_op(v):
            return adj_factor * K_sm_lr.T @ (K_sm_lr @ v) + self.lambd * (self.K_mm @ v)

        return K_inducing_sub_lin_op, K_inducing_sub_Kmm_lin_op
