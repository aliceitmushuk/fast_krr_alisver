import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import _get_kernel, _get_trace
from .model import Model


class InducingKRR(Model):
    def __init__(
        self,
        x,
        b,
        x_tst,
        b_tst,
        kernel_params,
        Knm_needed,
        inducing_pts,
        lambd,
        task,
        w0,
        device,
    ):
        super().__init__(x, b, x_tst, b_tst, kernel_params, lambd, task, w0, device)
        self.inducing_pts = inducing_pts
        self.m = self.inducing_pts.shape[0]
        self.bm_scaled = self.lambd / 2 * self.b[self.inducing_pts]
        self.inducing = True

        # Get inducing points kernel
        x_inducing_i = LazyTensor(self.x[self.inducing_pts][:, None, :])
        self.x_inducing_j = LazyTensor(self.x[self.inducing_pts][None, :, :])
        self.K_mm = _get_kernel(x_inducing_i, self.x_inducing_j, self.kernel_params)

        # Get kernel between full training set and inducing points
        if Knm_needed:
            x_i = LazyTensor(self.x[:, None, :])
            self.K_nm = _get_kernel(x_i, self.x_inducing_j, self.kernel_params)
            self.K_nmTb = self.K_nm.T @ self.b  # Useful for computing metrics
            self.K_nmTb_norm = torch.norm(self.K_nmTb)

        # Get kernel for test set
        x_tst_i = LazyTensor(self.x_tst[:, None, :])
        self.K_tst = _get_kernel(x_tst_i, self.x_inducing_j, self.kernel_params)

    def _Knm_lin_op(self, v):
        return self.K_nm @ v

    def _Kmm_lin_op(self, v):
        return self.K_mm @ v

    def lin_op(self, v):
        return self.K_nm.T @ self._Knm_lin_op(v) + self.lambd * self._Kmm_lin_op(v)

    def _compute_train_metrics(self, v):
        K_nmv = self._Knm_lin_op(v)
        K_mmv = self._Kmm_lin_op(v)
        residual = self.K_nm.T @ K_nmv + self.lambd * K_mmv - self.K_nmTb
        rel_residual = torch.norm(residual) / self.K_nmTb_norm
        loss = 1 / 2 * torch.norm(K_nmv - self.b) ** 2 + self.lambd / 2 * torch.dot(
            v, K_mmv
        )

        metrics_dict = {
            "rel_residual": rel_residual,
            "train_loss": loss,
        }

        return metrics_dict

    def _get_grad_regularizer(self):
        return -((self.lambd / 2) ** 2) * self.w + self.bm_scaled

    def _get_selection_idx(self, v1, v2):
        # Find elements of `v2` present in `v1` and their positions in `v1`
        common_elements_mask = torch.isin(v2, v1)
        common_elements_in_v2 = v2[common_elements_mask]

        # Get indices of `v1` corresponding to elements in `v2`
        indices_in_v1 = torch.where(torch.isin(v1, common_elements_in_v2))[0]

        # For each element of `v1` in `v2`, find its first occurrence in `v2`
        indices_in_v2 = torch.tensor(
            [torch.where(v2 == v1_elem)[0][0] for v1_elem in v1[indices_in_v1]]
        )

        return indices_in_v1, indices_in_v2

    def _get_table_aux(self, idx, w, table):
        # Find where the indices are in the inducing points
        indices_in_table, indices_in_w = self._get_selection_idx(self.inducing_pts, idx)
        w_selected = torch.zeros(idx.shape[0], device=self.device)
        # Select the appropriate elements of w to update weights
        # Be careful with the case where there are no common indices
        if torch.numel(indices_in_table) > 0 and torch.numel(indices_in_w) > 0:
            w_selected[indices_in_w] = w[indices_in_table]
        x_idx_i = LazyTensor(self.x[idx][:, None, :])
        K_nm_idx = _get_kernel(x_idx_i, self.x_inducing_j, self.kernel_params)
        new_weights = self.n * (
            K_nm_idx @ w + self.lambd / 2 * w_selected - self.b[idx]
        )

        weight_diff = new_weights - table[idx]
        weight_diff_selected = torch.zeros(self.m, device=self.device)

        # Update the weight difference for the indices in the inducing points
        # Be careful with the case where there are no common indices
        if torch.numel(indices_in_table) > 0 and torch.numel(indices_in_w) > 0:
            weight_diff_selected[indices_in_table] = weight_diff[indices_in_w]
        aux = K_nm_idx.T @ weight_diff + self.lambd / 2 * weight_diff_selected

        return new_weights, aux

    def _get_subsampled_lin_ops(self, bH, bH2):
        hess_pts = torch.randperm(self.n)[:bH]
        x_hess_i = LazyTensor(self.x[hess_pts][:, None, :])
        K_sm = _get_kernel(x_hess_i, self.x_inducing_j, self.kernel_params)

        hess_pts_lr = torch.randperm(self.n)[:bH2]
        x_hess_lr_i = LazyTensor(self.x[hess_pts_lr][:, None, :])
        K_sm_lr = _get_kernel(x_hess_lr_i, self.x_inducing_j, self.kernel_params)

        indices_in_inducing_pts, indices_in_hess_pts = self._get_selection_idx(
            self.inducing_pts, hess_pts
        )
        indices_in_inducing_pts_lr, indices_in_hess_pts_lr = self._get_selection_idx(
            self.inducing_pts, hess_pts_lr
        )

        # Defines matrix-vector product with unbiased estimate of the Hessian
        def K_inducing_sub_lin_op(
            v, K_sm, bH, indices_in_inducing_pts, indices_in_hess_pts
        ):
            adj_factor = self.n / bH
            K_sm_v = K_sm @ v

            if v.ndim == 1:
                v_selected = torch.zeros(bH, device=self.device)
                K_sm_v_selected = torch.zeros(self.m, device=self.device)
            elif v.ndim == 2:
                v_selected = torch.zeros(bH, v.shape[1], device=self.device)
                K_sm_v_selected = torch.zeros(self.m, v.shape[1], device=self.device)
            else:
                raise ValueError("v must be a 1D or 2D tensor")

            if (
                torch.numel(indices_in_inducing_pts) > 0
                and torch.numel(indices_in_hess_pts) > 0
            ):
                K_sm_v_selected[indices_in_inducing_pts] = K_sm_v[indices_in_hess_pts]
                v_selected[indices_in_hess_pts] = v[indices_in_inducing_pts]

            return adj_factor * (
                (K_sm.T @ K_sm_v)
                + self.lambd / 2 * (K_sm_v_selected + K_sm.T @ v_selected)
            )

        # Linear operator for calculating the preconditioner
        def K_inducing_precond_lin_op(v):
            return K_inducing_sub_lin_op(
                v, K_sm, bH, indices_in_inducing_pts, indices_in_hess_pts
            )

        # Linear operator for calculating the learning rate
        def K_inducing_lr_lin_op(v):
            return K_inducing_sub_lin_op(
                v, K_sm_lr, bH2, indices_in_inducing_pts_lr, indices_in_hess_pts_lr
            )

        # Trace of the subsampled Hessian
        # (based on the estimate used in the preconditioner)
        K_inducing_fro_norm2 = torch.sum(
            (K_sm**2).sum() @ torch.ones(1, device=self.device)
        ).item()
        K_inducing_trace = (
            self.n
            / bH
            * (
                K_inducing_fro_norm2
                + self.lambd
                * _get_trace(indices_in_inducing_pts.shape[0], self.kernel_params)
            )
        )

        return K_inducing_precond_lin_op, K_inducing_lr_lin_op, K_inducing_trace

    def _get_Kmm_trace(self):
        return _get_trace(self.m, self.kernel_params)
