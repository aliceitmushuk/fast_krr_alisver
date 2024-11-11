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
            if self.exp_args["bH2"] is None:
                self.exp_args["bH2"] = max(1, model.n // 50)

            if self.exp_args["opt"] == "sketchysvrg":
                if self.exp_args["update_freq"] is None:
                    self.exp_args["update_freq"] = model.n // self.exp_args["bg"]
            elif self.exp_args["opt"] == "sketchykatyusha":
                if self.exp_args["p"] is None:
                    self.exp_args["p"] = self.exp_args["bg"] / model.n
                if self.exp_args["mu"] is None:
                    self.exp_args["mu"] = model.lambd

    def _time_exceeded(self, n_iters, time_elapsed):
        if "max_time" in self.exp_args:
            if time_elapsed >= self.exp_args["max_time"]:
                return True
        if "max_iter" in self.exp_args:
            if n_iters >= self.exp_args["max_iter"]:
                return True
        return False

    def _get_eval_loc(self, model):
        return model.w

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
                not (self.exp_args["log_test_only"] and self.exp_args["opt"] != "pcg"),
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
                not (
                    self.exp_args["log_test_only"]
                    and self.exp_args["opt"] in ["sketchysgd", "sketchysaga"]
                ),
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
                eval_loc = self._get_eval_loc(model)

                logger.update_cum_time()

                # Always log metrics at the start
                logger.compute_log_reset(
                    -1, model.compute_metrics, eval_loc, config.log_test_only
                )

                # Terminate if max allowed time is exceeded
                if self._time_exceeded(0, logger.cum_time):
                    return

                i = 0  # Iteration counter

                # Run the optimizer
                while True:
                    opt.step()
                    eval_loc = self._get_eval_loc(model)

                    logger.update_cum_time()

                    # Terminate when max allowed time is exceeded
                    if self._time_exceeded(i + 1, logger.cum_time):
                        # Log the last iteration; we use -1 as a hack
                        logger.compute_log_reset(
                            -1, model.compute_metrics, eval_loc, config.log_test_only
                        )
                        return

                    logger.compute_log_reset(
                        i, model.compute_metrics, eval_loc, config.log_test_only
                    )

                    i += 1
