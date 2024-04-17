import argparse
import warnings

import wandb
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
from src.logger import Logger
from src.utils import ParseParams, set_random_seed, load_data

OPT_NAMES = {
    "skotch": Skotch,
    "askotch": ASkotch,
    "sketchysgd": SketchySGD,
    "sketchysvrg": SketchySVRG,
    "sketchysaga": SketchySAGA,
    "sketchykatyusha": SketchyKatyusha,
    "pcg": PCG,
}


def check_inputs(args):
    opt_name = OPT_NAMES[args.opt]
    # Input checking
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


def get_full_krr(Xtr, ytr, Xtst, ytst, kernel_params, lambd, task, device):
    w0 = torch.zeros(Xtr.shape[0], device=device)
    return FullKRR(Xtr, ytr, Xtst, ytst, kernel_params, lambd, task, w0, device)


def get_inducing_krr(Xtr, ytr, Xtst, ytst, kernel_params, m, lambd, task, device):
    w0 = torch.zeros(m, device=device)
    inducing_pts = torch.randperm(Xtr.shape[0])[:m]
    return InducingKRR(
        Xtr, ytr, Xtst, ytst, kernel_params, inducing_pts, lambd, task, w0, device
    )


def get_opt(model, config):
    if config.opt == "skotch":
        opt = Skotch(model, config.b, config.alpha, config.precond_params)
    elif config.opt == "askotch":
        opt = ASkotch(model, config.b, config.beta, config.precond_params)
    elif config.opt in [
        "sketchysgd",
        "sketchysvrg",
        "sketchysaga",
        "sketchykatyusha",
    ]:
        if config.opt == "sketchysgd":
            opt = SketchySGD(model, config.bg, config.bH, config.precond_params)
        elif config.opt == "sketchysvrg":
            opt = SketchySVRG(
                model, config.bg, config.bH, config.update_freq, config.precond_params
            )
        elif config.opt == "sketchysaga":
            opt = SketchySAGA(model, config.bg, config.bH, config.precond_params)
        elif config.opt == "sketchykatyusha":
            opt = SketchyKatyusha(
                model,
                config.bg,
                config.bH,
                config.p,
                config.lambd,
                config.precond_params,
            )
    elif config.opt == "pcg":
        opt = PCG(model, config.precond_params)

    return opt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="susy", help="Which dataset to use"
    )
    parser.add_argument(
        "--task", choices=["regression", "classification"], help="Type of task"
    )
    parser.add_argument(
        "--kernel_params",
        action=ParseParams,
        help='Kernel parameters in the form of a string: "type matern sigma 1.0 nu 1.5"',
    )
    parser.add_argument("--m", type=int, default=None, help="Number of inducing points")
    parser.add_argument(
        "--lambd", type=float, default=0.1, help="Regularization parameter"
    )
    parser.add_argument(
        "--opt",
        choices=[
            "skotch",
            "askotch",
            "sketchysgd",
            "sketchysvrg",
            "sketchysaga",
            "sketchykatyusha",
            "pcg",
        ],
        help="Which optimizer to use",
    )
    parser.add_argument(
        "--b", type=int, default=None, help="Number of blocks in optimizer"
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Sampling parameter in Skotch"
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Acceleration parameter in ASkotch"
    )
    parser.add_argument(
        "--bg", type=int, default=None, help="Gradient batch size in SGD-type methods"
    )
    parser.add_argument(
        "--bH", type=int, default=None, help="Hessian batch size in SGD-type methods"
    )
    parser.add_argument(
        "--update_freq", type=int, default=None, help="Update frequency in SketchySVRG"
    )
    parser.add_argument(
        "--p", type=float, default=None, help="Update probability in SketchyKatyusha"
    )
    parser.add_argument(
        "--precond_params",
        action=ParseParams,
        default=None,
        help='Preconditioner parameters in the form of a string: "type nystrom r 100 rho 0.1"',
    )
    parser.add_argument(
        "--max_iter", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--log_freq", type=int, default=100, help="Logging frequency of metrics"
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float32",
        help="Precision of the computations",
    )
    parser.add_argument("--seed", type=int, default=1234, help="initial seed")
    parser.add_argument("--device", type=str, default=0, help="GPU to use")
    parser.add_argument(
        "--wandb_project", type=str, default="fast_krr", help="W&B project name"
    )

    # Extract arguments from parser
    args = parser.parse_args()

    # Check the inputs
    check_inputs(args)

    # Set the precision
    set_precision(args.precision)

    # Set random seed
    set_random_seed(args.seed)

    # Organize arguments for the experiment into a dictionary for logging in wandb
    experiment_args = {
        "dataset": args.dataset,
        "task": args.task,
        "kernel_params": args.kernel_params,
        "lambd": args.lambd,
        "opt": args.opt,
        "precond_params": args.precond_params,
        "max_iter": args.max_iter,
        "log_freq": args.log_freq,
        "precision": args.precision,
        "seed": args.seed,
        "device": f"cuda:{args.device}",
    }

    if args.opt == "skotch":
        experiment_args["b"] = args.b
        experiment_args["alpha"] = args.alpha
        experiment_args["inducing"] = False
    elif args.opt == "askotch":
        experiment_args["b"] = args.b
        experiment_args["beta"] = args.beta
        experiment_args["inducing"] = False
    elif args.opt in ["sketchysgd", "sketchysvrg", "sketchysaga", "sketchykatyusha"]:
        experiment_args["m"] = args.m
        experiment_args["bg"] = args.bg
        experiment_args["bH"] = args.bH
        experiment_args["inducing"] = True

        if args.opt == "sketchysvrg":
            experiment_args["update_freq"] = args.update_freq
        elif args.opt == "sketchykatyusha":
            experiment_args["p"] = args.p
    elif args.opt == "pcg":
        if args.m is not None:
            experiment_args["m"] = args.m
            experiment_args["inducing"] = True
        else:
            experiment_args["inducing"] = False

    with wandb.init(project=args.wandb_project, config=experiment_args):
        # Access the experiment configuration
        config = wandb.config

        # Load the dataset
        Xtr, Xtst, ytr, ytst = load_data(config.dataset, config.seed, config.device)

        # Get the model, initializing at zero
        if config.inducing:
            model = get_inducing_krr(
                Xtr,
                ytr,
                Xtst,
                ytst,
                config.kernel_params,
                config.m,
                config.lambd,
                config.task,
                config.device,
            )
        else:
            model = get_full_krr(
                Xtr,
                ytr,
                Xtst,
                ytst,
                config.kernel_params,
                config.lambd,
                config.task,
                config.device,
            )

        # Select the optimizer
        opt = get_opt(model, config)

        # Initialize the logger
        logger = Logger(config.log_freq)

        # Run the optimizer
        with torch.no_grad():
            opt.run(config.max_iter, logger)


if __name__ == "__main__":
    main()
