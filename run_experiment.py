import argparse

from src.experiment import Experiment
from src.experiment_utils import (
    ParseParams,
    check_inputs,
    set_precision,
    set_random_seed,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="susy", help="Which dataset to use"
    )
    parser.add_argument(
        "--model", choices=["full_krr", "inducing_krr"], help="Type of model"
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
        "--bH2",
        type=int,
        default=None,
        help="Hessian batch size for eig calculations in SGD-type methods",
    )
    parser.add_argument(
        "--update_freq", type=int, default=None, help="Update frequency in SketchySVRG"
    )
    parser.add_argument(
        "--p", type=float, default=None, help="Update probability in SketchyKatyusha"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Strong convexity parameter in SketchyKatyusha",
    )
    parser.add_argument(
        "--precond_params",
        action=ParseParams,
        default=None,
        help='Preconditioner parameters in the form of a string: "type nystrom r 100 rho 0.1"',
    )
    parser.add_argument(
        "--max_iter", type=int, default=None, help="Number of iterations"
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=None,
        help="Maximum time (in seconds) to run the optimizer",
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
        "model": args.model,
        "task": args.task,
        "kernel_params": args.kernel_params,
        "lambd": args.lambd,
        "opt": args.opt,
        "precond_params": args.precond_params,
        "log_freq": args.log_freq,
        "precision": args.precision,
        "seed": args.seed,
        "device": f"cuda:{args.device}",
        "wandb_project": args.wandb_project,
    }

    if args.model == "inducing_krr":
        experiment_args["m"] = args.m

    if args.max_iter is not None:
        experiment_args["max_iter"] = args.max_iter
    if args.max_time is not None:
        experiment_args["max_time"] = args.max_time

    if args.opt == "skotch":
        experiment_args["b"] = args.b
        experiment_args["alpha"] = args.alpha
    elif args.opt == "askotch":
        experiment_args["b"] = args.b
        experiment_args["beta"] = args.beta
    elif args.opt in ["sketchysgd", "sketchysvrg", "sketchysaga", "sketchykatyusha"]:
        experiment_args["bg"] = args.bg
        experiment_args["bH"] = args.bH
        experiment_args["bH2"] = args.bH2

        if args.opt == "sketchysvrg":
            experiment_args["update_freq"] = args.update_freq
        elif args.opt == "sketchykatyusha":
            experiment_args["p"] = args.p
            experiment_args["mu"] = args.mu

    exp = Experiment(experiment_args)
    exp.run()


if __name__ == "__main__":
    main()
