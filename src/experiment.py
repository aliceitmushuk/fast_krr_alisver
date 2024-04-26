import torch
import wandb

from .experiment_utils import get_full_krr, get_inducing_krr, get_opt
from .data_utils import load_data
from .logger import Logger


class Experiment:
    def __init__(self, exp_args):
        self.exp_args = exp_args.copy()

    def _modify_opt_args(self, model):
        if self.exp_args["opt"].startswith("sketchy"):
            if self.exp_args["bH"] is None:
                self.exp_args["bH"] = int(model.n**0.5)

            if self.exp_args["opt"] == "sketchysvrg":
                if self.exp_args["update_freq"] is None:
                    self.exp_args["update_freq"] = model.n // self.exp_args["bg"]
            elif self.exp_args["opt"] == "sketchykatyusha":
                if self.exp_args["p"] is None:
                    self.exp_args["p"] = self.exp_args["bg"] / model.n
                if self.exp_args["mu"] is None:
                    self.exp_args["mu"] = model.lambd

    def _time_exceeded(self, time_elapsed):
        if "max_time" not in self.exp_args:
            return False
        return time_elapsed >= self.exp_args["max_time"]

    def run(self):
        # Load data
        Xtr, Xtst, ytr, ytst = load_data(
            self.exp_args["dataset"], self.exp_args["seed"], self.exp_args["device"]
        )

        # Load model
        if self.exp_args["model"] == "full_krr":
            model = get_full_krr(
                Xtr,
                ytr,
                Xtst,
                ytst,
                self.exp_args["kernel_params"],
                self.exp_args["lambd"],
                self.exp_args["task"],
                self.exp_args["device"],
            )
        elif self.exp_args["model"] == "inducing_krr":
            model = get_inducing_krr(
                Xtr,
                ytr,
                Xtst,
                ytst,
                self.exp_args["kernel_params"],
                self.exp_args["m"],
                self.exp_args["lambd"],
                self.exp_args["task"],
                self.exp_args["device"],
            )

        # Set optimizer args if needed
        self._modify_opt_args(model)

        with wandb.init(project=self.exp_args["wandb_project"], config=self.exp_args):
            # Access the wandb config
            config = wandb.config

            # Initialize logger
            logger = Logger(config.log_freq)

            with torch.no_grad():
                # Select and initialize the optimizer
                logger.reset_timer()
                opt = get_opt(model, config)
                if config.opt == "askotch":
                    eval_loc = opt.y
                else:
                    eval_loc = model.w
                logger.compute_log_reset(-1, model.compute_metrics, eval_loc)

                # Terminate if max allowed time is exceeded
                if self._time_exceeded(logger.cum_time):
                    return

                # Run the optimizer
                for i in range(config.max_iter):
                    opt.step()
                    if config.opt == "askotch":
                        eval_loc = opt.y
                    else:
                        eval_loc = model.w
                    logger.compute_log_reset(i, model.compute_metrics, eval_loc)

                    # Terminate if max allowed time is exceeded
                    if self._time_exceeded(logger.cum_time):
                        return
