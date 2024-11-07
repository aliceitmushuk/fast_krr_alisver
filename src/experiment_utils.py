import argparse
import random
import warnings

import numpy as np
import torch

from src.models import FullKRR, InducingKRR
from src.opts import (
    ASkotch,
    ASkotchV2,
    SketchySGD,
    SketchySVRG,
    SketchySAGA,
    SketchyKatyusha,
    PCG,
)

OPT_CLASSES = {
    "askotch": ASkotch,
    "askotchv2": ASkotchV2,
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


def check_required(arg, name, opt_name):
    if arg is None:
        raise ValueError(f"{name} must be provided for {opt_name}")


def check_askotch(args, opt_name):
    check_required(args.b, "Number of blocks", opt_name)
    check_required(args.beta, "Acceleration parameter", opt_name)
    check_required(args.accelerated, "Acceleration flag", opt_name)


def check_askotchv2(args, opt_name):
    check_required(args.block_sz, "Block size", opt_name)
    check_required(args.sampling_method, "Sampling method", opt_name)
    check_required(args.mu, "Mu", opt_name)
    check_required(args.nu, "Nu", opt_name)
    check_required(args.accelerated, "Acceleration flag", opt_name)


def check_sketchy(args, opt_name):
    check_required(args.m, "Number of inducing points", opt_name)
    check_required(args.bg, "Gradient batch size", opt_name)
    if args.bH is None:
        warnings.warn(
            f"Hessian batch size is not provided for {opt_name}. \
                Using default value int(n**0.5)"
        )
    if args.bH2 is None:
        warnings.warn(
            f"Hessian batch size for eig calculations is not provided for {opt_name}. \
                Using default value max(1, n // 50)"
        )
    if args.update_freq is not None and args.opt != "sketchysvrg":
        warnings.warn(
            f"Update frequency is not used in {opt_name}. Ignoring this parameter"
        )
    if args.p is None and args.opt == "sketchykatyusha":
        warnings.warn(
            f"Update probability is not provided for {opt_name}. \
                Using default value bg/n"
        )
    if args.mu is None and args.opt == "sketchykatyusha":
        warnings.warn(
            f"Strong convexity parameter is not provided for {opt_name}. \
                Using default value lambd"
        )


def check_pcg(args, opt_name):
    check_required(args.precond_params, "Preconditioner parameters", opt_name)


def check_precond_params(precond_params):
    if "type" not in precond_params:
        raise ValueError("Preconditioner type must be provided")
    if precond_params["type"] not in ["nystrom", "partial_cholesky", "falkon"]:
        raise ValueError(
            "Only Nystrom, Partial Cholesky, and Falkon preconditioners are supported"
        )


def check_inputs(args):
    if "max_time" not in args and "max_iter" not in args:
        raise ValueError("At least one of max_time or max_iter must be provided")

    opt_name = OPT_CLASSES[args.opt]
    opt_checkers = {
        "askotch": check_askotch,
        "askotchv2": check_askotchv2,
        "sketchysgd": check_sketchy,
        "sketchysvrg": check_sketchy,
        "sketchysaga": check_sketchy,
        "sketchykatyusha": check_sketchy,
        "pcg": check_pcg,
    }

    # Call the appropriate function based on optimizer type
    if args.opt in opt_checkers:
        opt_checkers[args.opt](args, opt_name)

    if args.precond_params is not None:
        check_precond_params(args.precond_params)


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
    return FullKRR(
        x=Xtr,
        b=ytr,
        x_tst=Xtst,
        b_tst=ytst,
        kernel_params=kernel_params,
        Ktr_needed=Ktr_needed,
        lambd=lambd,
        task=task,
        w0=w0,
        device=device,
    )


def get_inducing_krr(
    Xtr, ytr, Xtst, ytst, kernel_params, Knm_needed, m, lambd, task, device
):
    w0 = torch.zeros(m, device=device)
    inducing_pts = torch.randperm(Xtr.shape[0])[:m]
    return InducingKRR(
        x=Xtr,
        b=ytr,
        x_tst=Xtst,
        b_tst=ytst,
        kernel_params=kernel_params,
        Knm_neeeded=Knm_needed,
        inducing_pts=inducing_pts,
        lambd=lambd,
        task=task,
        w0=w0,
        device=device,
    )


def build_opt_params(model, config):
    if config.opt == "askotch":
        return {
            "model": model,
            "B": config.b,
            "no_store_precond": config.no_store_precond,
            "precond_params": config.precond_params,
            "beta": config.beta,
            "accelerated": config.accelerated,
        }
    elif config.opt == "askotchv2":
        return {
            "model": model,
            "block_sz": config.block_sz,
            "sampling_method": config.sampling_method,
            "precond_params": config.precond_params,
            "mu": config.mu,
            "nu": config.nu,
            "accelerated": config.accelerated,
        }
    elif config.opt == "sketchysgd":
        return {
            "model": model,
            "bg": config.bg,
            "bH": config.bH,
            "bH2": config.bH2,
            "precond_params": config.precond_params,
        }
    elif config.opt == "sketchysvrg":
        return {
            "model": model,
            "bg": config.bg,
            "bH": config.bH,
            "bH2": config.bH2,
            "update_freq": config.update_freq,
            "precond_params": config.precond_params,
        }
    elif config.opt == "sketchysaga":
        return {
            "model": model,
            "bg": config.bg,
            "bH": config.bH,
            "bH2": config.bH2,
            "precond_params": config.precond_params,
        }
    elif config.opt == "sketchykatyusha":
        return {
            "model": model,
            "bg": config.bg,
            "bH": config.bH,
            "bH2": config.bH2,
            "p": config.p,
            "mu": config.mu,
            "precond_params": config.precond_params,
        }
    elif config.opt == "pcg":
        return {
            "model": model,
            "precond_params": config.precond_params,
        }


def get_opt(model, config):
    # Build the parameter dictionary for the specified optimizer
    opt_params = build_opt_params(model, config)
    # Initialize the optimizer with the specified parameters
    return OPT_CLASSES[config.opt](**opt_params)
