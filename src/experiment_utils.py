import argparse
import random
import warnings

import numpy as np
import torch

from src.models.full_krr import FullKRR
from src.models.inducing_krr import InducingKRR
from src.opts.skotch import Skotch
from src.opts.askotch import ASkotch
from src.opts.sketchysgd import SketchySGD
from src.opts.sketchysvrg import SketchySVRG
from src.opts.sketchysaga import SketchySAGA
from src.opts.sketchykatyusha import SketchyKatyusha
from src.opts.pcg import PCG

OPT_NAMES = {
    "skotch": Skotch,
    "askotch": ASkotch,
    "sketchysgd": SketchySGD,
    "sketchysvrg": SketchySVRG,
    "sketchysaga": SketchySAGA,
    "sketchykatyusha": SketchyKatyusha,
    "pcg": PCG,
}


# Custom action to parse parameters
class ParseParams(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Split the input string into individual elements
        elements = values.split()
        # print(elements) # Useful for debugging
        params_dict = {}
        # Iterate over the elements two at a time (key-value pairs)
        i = 0
        while i < len(elements):
            key = elements[i]
            if key == "use_cpu":
                # Special case for the boolean key
                params_dict[key] = True
                i += 1
            else:
                value = elements[i + 1]
                # Attempt to convert numeric values to float, otherwise keep as string
                try:
                    if key == "r":  # Rank parameter in preconditioner is int, not float
                        value = int(value)
                    else:
                        value = float(value)
                except ValueError:
                    # If conversion fails, value remains a string
                    pass
                params_dict[key] = value
                i += 2
        setattr(namespace, self.dest, params_dict)


def check_inputs(args):
    # Check that at least one of max_time or max_iter is provided
    if "max_time" not in args and "max_iter" not in args:
        raise ValueError("At least one of max_time or max_iter must be provided")

    opt_name = OPT_NAMES[args.opt]
    # Input checking for optimizers
    if args.opt == "skotch":
        if args.m is not None:
            warnings.warn(
                f"Number of inducing points is not used in {opt_name}. Ignoring this parameter"
            )
        if args.b is None:
            raise ValueError(f"Number of blocks must be provided for {opt_name}")
        if args.alpha is None:
            raise ValueError(f"Sampling parameter must be provided for {opt_name}")
        if args.beta is not None:
            warnings.warn(f"Beta is not used in {opt_name}. Ignoring this parameter")
        if args.bg is not None:
            warnings.warn(
                f"Gradient batch size is not used in {opt_name}. Ignoring this parameter"
            )
        if args.bH is not None:
            warnings.warn(
                f"Hessian batch size is not used in {opt_name}. Ignoring this parameter"
            )
        if args.bH2 is not None:
            warnings.warn(
                f"Hessian batch size for eig calculations is not used in {opt_name}. Ignoring this parameter"
            )
    elif args.opt == "askotch":
        if args.m is not None:
            warnings.warn(
                f"Number of inducing points is not used in {opt_name}. Ignoring this parameter"
            )
        if args.b is None:
            raise ValueError(f"Number of blocks must be provided for {opt_name}")
        if args.alpha is not None:
            warnings.warn(
                f"Sampling parameter is not used in {opt_name}. Ignoring this parameter"
            )
        if args.beta is None:
            raise ValueError(f"Acceleration parameter must be provided for {opt_name}")
        if args.bg is not None:
            warnings.warn(
                f"Gradient batch size is not used in {opt_name}. Ignoring this parameter"
            )
        if args.bH is not None:
            warnings.warn(
                f"Hessian batch size is not used in {opt_name}. Ignoring this parameter"
            )
        if args.bH2 is not None:
            warnings.warn(
                f"Hessian batch size for eig calculations is not used in {opt_name}. Ignoring this parameter"
            )
    elif args.opt in ["sketchysgd", "sketchysvrg", "sketchysaga", "sketchykatyusha"]:
        if args.m is None:
            raise ValueError(
                f"Number of inducing points must be provided for {opt_name}"
            )
        if args.b is not None:
            warnings.warn(
                f"Number of blocks is not used in {opt_name}. Ignoring this parameter"
            )
        if args.alpha is not None:
            warnings.warn(
                f"Sampling parameter is not used in {opt_name}. Ignoring this parameter"
            )
        if args.beta is not None:
            warnings.warn(f"Beta is not used in {opt_name}. Ignoring this parameter")
        if args.bg is None:
            raise ValueError(f"Gradient batch size must be provided for {opt_name}")
        if args.bH is None:
            warnings.warn(
                f"Hessian batch size is not provided for {opt_name}. Using default value int(n**0.5)"
            )
        if args.bH2 is None:
            warnings.warn(
                f"Hessian batch size for eig calculations is not provided for {opt_name}. Using default value max(1, n // 50)"
            )

        if args.opt in ["sketchysgd", "sketchysaga", "sketchykatyusha"]:
            if args.update_freq is not None:
                warnings.warn(
                    f"Update frequency is not used in {opt_name}. Ignoring this parameter"
                )
        elif args.opt == "sketchysvrg":
            if args.update_freq is None:
                warnings.warn(
                    f"Update frequency is not provided for {opt_name}. Using default value n // bg"
                )

        if args.opt in ["sketchysgd", "sketchysvrg", "sketchysaga"]:
            if args.p is not None:
                warnings.warn(
                    f"Update probability is not used in {opt_name}. Ignoring this parameter"
                )
        elif args.opt == "sketchykatyusha":
            if args.p is None:
                warnings.warn(
                    f"Update probability is not provided for {opt_name}. Using default value bg/n"
                )

        if args.opt in ["sketchysgd", "sketchysvrg", "sketchysaga"]:
            if args.mu is not None:
                warnings.warn(
                    f"Strong convexity parameter is not used in {opt_name}. Ignoring this parameter"
                )
        elif args.opt == "sketchykatyusha":
            if args.mu is None:
                warnings.warn(
                    f"Strong convexity parameter is not provided for {opt_name}. Using default value lambd"
                )
    elif args.opt == "pcg":
        if args.b is not None:
            warnings.warn(
                f"Number of blocks is not used in {opt_name}. Ignoring this parameter"
            )
        if args.alpha is not None:
            warnings.warn(
                f"Sampling parameter is not used in {opt_name}. Ignoring this parameter"
            )
        if args.beta is not None:
            warnings.warn(f"Beta is not used in {opt_name}. Ignoring this parameter")
        if args.bg is not None:
            warnings.warn(
                f"Gradient batch size is not used in {opt_name}. Ignoring this parameter"
            )
        if args.bH is not None:
            warnings.warn(
                f"Hessian batch size is not used in {opt_name}. Ignoring this parameter"
            )
        if args.bH2 is not None:
            warnings.warn(
                f"Hessian batch size for eig calculations is not used in {opt_name}. Ignoring this parameter"
            )
        if args.update_freq is not None:
            warnings.warn(
                f"Update frequency is not used in {opt_name}. Ignoring this parameter"
            )
        if args.p is not None:
            warnings.warn(
                f"Update probability is not used in {opt_name}. Ignoring this parameter"
            )

        if args.precond_params is None:
            raise ValueError(
                f"Preconditioner parameters must be provided for {opt_name}"
            )

    # TODO: Improve this
    if args.precond_params is not None:
        # Check that 'type' is provided and 'nystrom' is the only option
        if "type" not in args.precond_params:
            raise ValueError("Preconditioner type must be provided")
        if args.precond_params["type"] not in ["nystrom", "partial_cholesky", "falkon"]:
            raise ValueError(
                "Only Nystrom, Partial Cholesky, and Falkon preconditioners are supported"
            )

        # TODO: Check that the required parameters are provided for Nystrom.
        # Note that rho is not required for Skotch/A-Skotch but is required for PROMISE methods


def set_precision(precision):
    if precision == "float32":
        torch.set_default_dtype(torch.float32)
    elif precision == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError("Precision must be either 'float32' or 'float64'")


"""
Helper function for setting seed for the random number generator in various packages.

INPUT: 
- seed: integer
"""


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_full_krr(Xtr, ytr, Xtst, ytst, kernel_params, Ktr_needed, lambd, task, device):
    w0 = torch.zeros(Xtr.shape[0], device=device)
    return FullKRR(Xtr, ytr, Xtst, ytst, kernel_params, Ktr_needed, lambd, task, w0, device)


def get_inducing_krr(Xtr, ytr, Xtst, ytst, kernel_params, Knm_needed, m, lambd, task, device):
    w0 = torch.zeros(m, device=device)
    inducing_pts = torch.randperm(Xtr.shape[0])[:m]
    return InducingKRR(
        Xtr, ytr, Xtst, ytst, kernel_params, Knm_needed, inducing_pts, lambd, task, w0, device
    )


def get_opt(model, config):
    if config.opt == "skotch":
        opt = Skotch(model, config.b, config.no_store_precond, config.alpha, config.precond_params)
    elif config.opt == "askotch":
        opt = ASkotch(model, config.b, config.no_store_precond, config.beta, config.precond_params)
    elif config.opt in [
        "sketchysgd",
        "sketchysvrg",
        "sketchysaga",
        "sketchykatyusha",
    ]:
        if config.opt == "sketchysgd":
            opt = SketchySGD(
                model, config.bg, config.bH, config.bH2, config.precond_params
            )
        elif config.opt == "sketchysvrg":
            opt = SketchySVRG(
                model,
                config.bg,
                config.bH,
                config.bH2,
                config.update_freq,
                config.precond_params,
            )
        elif config.opt == "sketchysaga":
            opt = SketchySAGA(
                model, config.bg, config.bH, config.bH2, config.precond_params
            )
        elif config.opt == "sketchykatyusha":
            opt = SketchyKatyusha(
                model,
                config.bg,
                config.bH,
                config.bH2,
                config.p,
                config.mu,
                config.precond_params,
            )
    elif config.opt == "pcg":
        opt = PCG(model, config.precond_params)

    return opt
