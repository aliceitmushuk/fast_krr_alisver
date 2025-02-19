from abc import ABC, abstractmethod
from typing import Dict, Union

from pykeops.torch import LazyTensor
import torch

from fast_krr.kernels.kernel_inits import (
    _get_kernel,
    _get_trace,
    _get_row,
)


class Model(ABC):
    """
    Abstract base class for Kernel Ridge Regression (KRR) models.

    This class provides shared functionality for KRR models
    such as FullKRR and InducingKRR.
    Subclasses must implement the `lin_op` and `_compute_train_metrics` methods.

    Attributes:
        x (torch.Tensor): Training features of shape (n_samples, n_features).
        b (torch.Tensor): Training targets of shape (n_samples,).
        x_tst (torch.Tensor): Testing features of shape (n_test_samples, n_features).
        b_tst (torch.Tensor): Testing targets of shape (n_test_samples,).
        kernel_params (Dict[str, float]): Parameters for the kernel function.
        lambd (float): Regularization parameter.
        task (str): Task type, either "classification" or "regression".
        w (torch.Tensor): Model weights, initialized as `w0`.
        device (torch.device): Device on which to perform computations.
    """

    def __init__(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        x_tst: torch.Tensor,
        b_tst: torch.Tensor,
        kernel_params: Dict[str, Union[str, float]],
        lambd: float,
        task: str,
        w0: torch.Tensor,
        device: torch.device,
    ) -> None:
        """
        Initialize the Model.

        Args:
            x (torch.Tensor): Training features of shape (n_samples, n_features).
            b (torch.Tensor): Training targets of shape (n_samples,).
            x_tst (torch.Tensor): Testing features of shape (n_tst_samples, n_features).
            b_tst (torch.Tensor): Testing targets of shape (n_tst_samples,).
            kernel_params (Dict[str, float]): Parameters for the kernel function.
            lambd (float): Regularization parameter.
            task (str): Task type, either "classification" or "regression".
            w0 (torch.Tensor): Initial model weights.
            device (torch.device): Device for computation.
        """
        self.x = x
        self.b = b
        self.x_tst = x_tst
        self.b_tst = b_tst
        self.kernel_params = kernel_params
        self.lambd = lambd
        self.task = task
        self.w = w0
        self.device = device

        self.b_norm = torch.norm(self.b)
        self.n = self.x.shape[0]
        self.n_tst = self.x_tst.shape[0] if self.x_tst is not None else 0

        self.K_tst = None  # To be set in subclass

    @abstractmethod
    def lin_op(self, v: torch.Tensor) -> torch.Tensor:
        """
        Linear operator specific to the model.

        Args:
            v (torch.Tensor): Input vector to apply the operator to.

        Returns:
            torch.Tensor: Result of the linear operation.
        """
        pass

    @abstractmethod
    def _compute_train_metrics(self, v: torch.Tensor) -> Dict[str, float]:
        """
        Compute training metrics specific to the model.

        Args:
            v (torch.Tensor): Model weights or coefficients.

        Returns:
            Dict[str, float]: Dictionary of training metrics.
        """
        pass

    def _compute_test_metrics(self, v: torch.Tensor) -> Dict[str, float]:
        """
        Compute test metrics common to all models.

        Args:
            v (torch.Tensor): Model weights or coefficients.

        Returns:
            Dict[str, float]: Dictionary of test metrics.
        """
        metrics_dict: Dict[str, float] = {}
        pred = self.K_tst @ v
        if self.task == "classification":
            metrics_dict["test_acc"] = float(
                torch.sum(torch.sign(pred) == self.b_tst) / self.n_tst
            )
        else:
            metrics_dict["test_mse"] = float(
                1 / 2 * torch.norm(pred - self.b_tst) ** 2 / self.n_tst
            )
            metrics_dict["test_msre"] = float(
                1 / 2 * torch.norm(pred / self.b_tst - 1.0) ** 2 / self.n_tst
            )
            metrics_dict["test_rmse"] = metrics_dict["test_mse"] ** 0.5
            metrics_dict["test_rmsre"] = metrics_dict["test_msre"] ** 0.5
            abs_errs = (pred - self.b_tst).abs()
            metrics_dict["test_smape"] = float(
                torch.sum(abs_errs / ((pred.abs() + self.b_tst.abs()) / 2)) / self.n_tst
            )
            metrics_dict["test_mae"] = float(abs_errs.mean())
        return metrics_dict

    def compute_metrics(self, v: torch.Tensor, log_test_only: bool) -> Dict[str, float]:
        """
        Compute metrics for the model.

        Combines training and testing metrics depending on the `log_test_only` flag.

        Args:
            v (torch.Tensor): Model weights or coefficients.
            log_test_only (bool): If True, only compute test metrics.

        Returns:
            Dict[str, float]: Combined dictionary of metrics.
        """
        metrics_dict: Dict[str, float] = {}
        if not log_test_only:
            metrics_dict.update(self._compute_train_metrics(v))
        metrics_dict.update(self._compute_test_metrics(v))
        return metrics_dict

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

    def _get_kernel_fn(self):
        def K_fn(x_i, x_j, get_row):
            if get_row:
                return _get_row(x_i, x_j, self.kernel_params)  # Tensor
            else:
                x_i_lz = LazyTensor(x_i[:, None, :])
                x_j_lz = LazyTensor(x_j[None, :, :])
                return _get_kernel(x_i_lz, x_j_lz, self.kernel_params)  # LazyTensor

        return K_fn
