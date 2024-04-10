import time
import torch
import wandb


class Logger:
    def __init__(self, log_freq):
        self.log_freq = log_freq
        self.start_time = time.time()
        self.iter_time = 0

    def compute_metrics(self, K_lin_op, K_tst, a, b, b_tst, b_org_norm, task, inducing):
        b_norm = torch.norm(
            b
        )  # Accounts for modifications to b in the case of inducing points

        residual = K_lin_op(a) - b
        rel_residual = torch.norm(residual) / b_norm
        if inducing:
            loss = 1 / 2 * (torch.dot(a, residual - b) + b_org_norm**2)
        else:
            loss = 1 / 2 * torch.dot(a, residual - b)

        metrics_dict = {"rel_residual": rel_residual, "train_loss": loss}

        pred = K_tst @ a

        test_metric_name = "test_acc" if task == "classification" else "test_mse"
        if task == "classification":
            test_metric = torch.sum(torch.sign(pred) == b_tst) / b_tst.shape[0]
            metrics_dict[test_metric_name] = test_metric
        else:
            test_metric = 1 / 2 * torch.norm(pred - b_tst) ** 2 / b_tst.shape[0]
            smape = (
                torch.sum((pred - b_tst).abs() / ((pred.abs() + b_tst.abs()) / 2))
                / b_tst.shape[0]
            )
            metrics_dict[test_metric_name] = test_metric
            metrics_dict["smape"] = smape

        return metrics_dict

    def reset_timer(self):
        self.start_time = time.time()
        self.iter_time = 0

    def compute_log_reset(
        self, K_lin_op, K_tst, a, b, b_tst, b_org_norm, task, i, inducing
    ):
        self.iter_time = time.time() - self.start_time

        metrics_dict = {}
        if (i + 1) % self.log_freq == 0:
            metrics_dict = self.compute_metrics(
                K_lin_op, K_tst, a, b, b_tst, b_org_norm, task, inducing
            )

        wandb.log({"iter_time": self.iter_time} | metrics_dict)

        self.reset_timer()
