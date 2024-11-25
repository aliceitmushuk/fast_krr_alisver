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
        self.test_metric_name = (
            "test_acc" if self.task == "classification" else "test_mse"
        )

    @abstractmethod
    def lin_op(self, v):
        pass

    @abstractmethod
    def compute_metrics(self, v, log_test_only):
        pass
