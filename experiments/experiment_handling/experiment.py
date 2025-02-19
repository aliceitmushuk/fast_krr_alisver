import torch
import wandb

from experiment_handling.utils import (
    get_full_krr,
    get_inducing_krr,
    get_opt,
    get_bandwidth,
)
from data_handling.utils import load_data
from logger import Logger

N_PAIRS = 1000  # Number of samples for median heuristic in kernel bandwidth computation


class Experiment:
    def __init__(self, exp_args):
        self.exp_args = exp_args.copy()

    def _modify_exp_args(self, Xtr):
        # Model parameters
        self.exp_args["n"] = Xtr.shape[0]
        self.exp_args["lambd"] = self.exp_args["lambd_unscaled"] * self.exp_args["n"]

        # Kernel parameters
        self.exp_args["kernel_params"]["sigma"] = get_bandwidth(
            Xtr, self.exp_args["kernel_params"]["sigma"], N_PAIRS
        )

        # ASkotchV2 parameters
        if "block_sz_frac" in self.exp_args:
            self.exp_args["block_sz"] = int(
                self.exp_args["block_sz_frac"] * self.exp_args["n"]
            )
        if "mu" in self.exp_args and self.exp_args["mu"] is None:
            self.exp_args["mu"] = self.exp_args["lambd"]
        if "nu" in self.exp_args and self.exp_args["nu"] is None:
            self.exp_args["nu"] = self.exp_args["n"] / self.exp_args["block_sz"]

        # EigenPro2 parameters
        if "gamma" in self.exp_args and self.exp_args["gamma"] is None:
            self.exp_args["gamma"] = 0.95

        # EigenPro3 parameters
        if (
            "proj_inner_iters" in self.exp_args
            and self.exp_args["proj_inner_iters"] is None
        ):
            self.exp_args["proj_inner_iters"] = 10

        # Mimosa parameters
        if "bH" in self.exp_args and self.exp_args["bH"] is None:
            self.exp_args["bH"] = int(self.exp_args["n"] ** 0.5)
        if "bH2" in self.exp_args and self.exp_args["bH2"] is None:
            self.exp_args["bH2"] = max(1, self.exp_args["n"] // 50)

    # Accounts for EigenPro methods automatically setting bg based on the data
    def _update_eigenpro_config(self, config, opt):
        if config.opt in ["eigenpro2", "eigenpro3"]:
            config.update({"bg": opt.bg}, allow_val_change=True)
            config.update({"eta": opt.eta}, allow_val_change=True)

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

        # Set optimizer args if needed
        self._modify_exp_args(Xtr)

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
            if self.exp_args["m"] > self.exp_args["n"]:
                raise ValueError(
                    "Number of inducing points must be less than number of data points"
                )
            model = get_inducing_krr(
                Xtr,
                ytr,
                Xtst,
                ytst,
                self.exp_args["kernel_params"],
                not (self.exp_args["log_test_only"] and self.exp_args["opt"] != "pcg"),
                self.exp_args["m"],
                self.exp_args["lambd"],
                self.exp_args["task"],
                self.exp_args["device"],
            )

        with wandb.init(project=self.exp_args["wandb_project"], config=self.exp_args):
            # Access the wandb config
            config = wandb.config

            # Initialize logger
            logger = Logger(config.log_freq)

            with torch.no_grad():
                # Select and initialize the optimizer
                logger.reset_timer()
                opt = get_opt(model, config)
                self._update_eigenpro_config(config, opt)
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
        wandb.finish()
