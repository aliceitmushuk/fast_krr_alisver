from abc import ABC, abstractmethod

import torch


class Model(ABC):
    def __init__(self, x, b, x_tst, b_tst, kernel_params, lambd, task, w0, device):
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
        self.n_tst = self.x_tst.shape[0]

        self.K_tst = None

    @abstractmethod
    def lin_op(self, v):
        pass

    @abstractmethod
    def _compute_train_metrics(self, v):
        pass

    def _compute_test_metrics(self, v):
        metrics_dict = {}
        pred = self.K_tst @ v
        if self.task == "classification":
            metrics_dict["test_acc"] = (
                torch.sum(torch.sign(pred) == self.b_tst) / self.n_tst
            )
        else:
            metrics_dict["test_mse"] = (
                1 / 2 * torch.norm(pred - self.b_tst) ** 2 / self.n_tst
            )
            metrics_dict["test_rmse"] = metrics_dict["test_mse"] ** 0.5
            abs_errs = (pred - self.b_tst).abs()
            metrics_dict["test_smape"] = (
                torch.sum(abs_errs / ((pred.abs() + self.b_tst.abs()) / 2)) / self.n_tst
            )
            metrics_dict["test_mae"] = abs_errs.mean()
        return metrics_dict

    def compute_metrics(self, v, log_test_only):
        metrics_dict = {}
        if not log_test_only:
            metrics_dict.update(self._compute_train_metrics(v))
        metrics_dict.update(self._compute_test_metrics(v))

        return metrics_dict
