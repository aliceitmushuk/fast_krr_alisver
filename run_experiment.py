import wandb
import argparse
import torch

from src.opts.skotch import Skotch
from src.opts.askotch import ASkotch
from src.opts.sketchysgd import SketchySGD
from src.opts.sketchysvrg import SketchySVRG
from src.opts.sketchysaga import SketchySAGA
from src.logger import Logger
from src.utils import ParseParams, set_random_seed, load_data

OPT_NAMES = {'skotch': Skotch, 'askotch': ASkotch, 
            'sketchysgd': SketchySGD, 'sketchysvrg': SketchySVRG, 'sketchysaga': SketchySAGA}

def check_inputs(args):
    opt_name = OPT_NAMES[args.opt]
    # Input checking
    if args.opt == 'skotch':
        if args.m is not None:
            raise Warning(
                f'Number of inducing points is not used in {opt_name}. Ignoring this parameter')
        if args.b is None:
            raise ValueError(
                f'Number of blocks must be provided for {opt_name}')
        if args.beta is not None:
            raise Warning(
                f'Beta is not used in {opt_name}. Ignoring this parameter')
        if args.bg is not None:
            raise Warning(
                f'Gradient batch size is not used in {opt_name}. Ignoring this parameter')
        if args.bH is not None:
            raise Warning(
                f'Hessian batch size is not used in {opt_name}. Ignoring this parameter')
    elif args.opt == 'askotch':
        if args.m is not None:
            raise Warning(
                f'Number of inducing points is not used in {opt_name}. Ignoring this parameter')
        if args.b is None:
            raise ValueError(
                f'Number of blocks must be provided for {opt_name}')
        if args.beta is None:
            raise ValueError(f'Beta must be provided for {opt_name}')
        if args.bg is not None:
            raise Warning(
                f'Gradient batch size is not used in {opt_name}. Ignoring this parameter')
        if args.bH is not None:
            raise Warning(
                f'Hessian batch size is not used in {opt_name}. Ignoring this parameter')
    elif args.opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
        if args.m is None:
            raise ValueError(
                f'Number of inducing points must be provided for {opt_name}')
        if args.b is not None:
            raise Warning(
                f'Number of blocks is not used in {opt_name}. Ignoring this parameter')
        if args.beta is not None:
            raise Warning(
                f'Beta is not used in {opt_name}. Ignoring this parameter')
        if args.bg is None:
            raise ValueError(
                f'Gradient batch size must be provided for {opt_name}')
        if args.bH is None:
            raise ValueError(
                f'Hessian batch size must be provided for {opt_name}')
        
        if args.opt in ['sketchysgd', 'sketchysaga']:
            if args.update_freq is not None:
                raise Warning(
                    f'Update frequency is not used in {opt_name}. Ignoring this parameter')
        elif args.opt == 'sketchysvrg':
            if args.update_freq is None:
                raise ValueError(
                    f'Update frequency must be provided for {opt_name}')
        
    if args.precond_params is not None:
        # Check that 'type' is provided and 'nystrom' is the only option
        if 'type' not in args.precond_params:
            raise ValueError('Preconditioner type must be provided')
        if args.precond_params['type'] != 'nystrom':
            raise ValueError('Only Nystrom preconditioner is supported')

        # TODO: Check that the required parameters are provided for Nystrom.
        # Note that rho is not required for Skotch/A-Skotch but is required for PROMISE methods


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='susy', help='Which dataset to use')
    parser.add_argument('--task', choices=['regression', 'classification'], help='Type of task')
    parser.add_argument('--kernel_params', action=ParseParams, 
        help='Kernel parameters in the form of a string: "type matern sigma 1.0 nu 1.5"')
    parser.add_argument('--m', type=int, default=None, help='Number of inducing points')
    parser.add_argument('--lambd', type=float, default=0.1, help='Regularization parameter')
    parser.add_argument('--opt', choices=['skotch', 'askotch', 'sketchysgd', 'sketchysvrg', 'sketchysaga'], help='Which optimizer to use')
    parser.add_argument('--b', type=int, default=None, help='Number of blocks in optimizer')
    parser.add_argument('--beta', type=float, default=None, help='Acceleration parameter in ASkotch')
    parser.add_argument('--bg', type=int, default=None, help='Gradient batch size in SGD-type methods')
    parser.add_argument('--bH', type=int, default=None, help='Hessian batch size in SGD-type methods')
    parser.add_argument('--update_freq', type=int, default=None, help='Update frequency in SketchySVRG')
    parser.add_argument('--precond_params', action=ParseParams, default=None,
        help='Preconditioner parameters in the form of a string: "type nystrom r 100 rho 0.1"')
    parser.add_argument('--max_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency of metrics')
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--wandb_project', type=str, default='fast_krr', help='W&B project name')

    # Extract arguments from parser
    args = parser.parse_args()

    # Check the inputs
    check_inputs(args)

    # Set random seed
    seed = args.seed
    set_random_seed(seed)

    # Organize arguments for the experiment into a dictionary for logging in wandb
    experiment_args = {
        'dataset': args.dataset,
        'task': args.task,
        'kernel_params': args.kernel_params,
        'lambd': args.lambd,
        'opt': args.opt,
        'precond_params': args.precond_params,
        'max_iter': args.max_iter,
        'log_freq': args.log_freq,
        'seed': seed,
        'device': f'cuda:{args.device}'
    }

    if args.opt == 'skotch':
        experiment_args['b'] = args.b
    elif args.opt == 'askotch':
        experiment_args['b'] = args.b
        experiment_args['beta'] = args.beta
    elif args.opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
        experiment_args['m'] = args.m
        experiment_args['bg'] = args.bg
        experiment_args['bH'] = args.bH

        if args.opt == 'sketchysvrg':
            experiment_args['update_freq'] = args.update_freq

    with wandb.init(project=args.wandb_project, config=experiment_args):
        # Access the experiment configuration
        config = wandb.config

        # Load the dataset
        Xtr, Xtst, ytr, ytst = load_data(config.dataset, config.seed, config.device)

        # Select the optimizer
        if config.opt == 'skotch':
            opt = Skotch(config.b, config.precond_params)
        elif config.opt == 'askotch':
            opt = ASkotch(config.b, config.beta, config.precond_params)
        elif config.opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
            inducing_pts = torch.randperm(Xtr.shape[0])[:config.m]

            if config.opt == 'sketchysgd':
                opt = SketchySGD(config.bg, config.bH, config.precond_params)
            elif config.opt == 'sketchysvrg':
                opt = SketchySVRG(config.bg, config.bH, config.update_freq,
                                   config.precond_params)
            elif config.opt == 'sketchysaga':
                opt = SketchySAGA(config.bg, config.bH, config.precond_params)

        # Initialize at 0
        if config.opt == 'skotch' or config.opt == 'askotch':
            a0 = torch.zeros(Xtr.shape[0], device=config.device)
        elif config.opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
            a0 = torch.zeros(config.m, device=config.device)

        # Initialize the logger
        logger = Logger(config.log_freq)

        # Run the optimizer
        with torch.no_grad():
            if config.opt in ['sketchysgd', 'sketchysvrg', 'sketchysaga']:
                opt.run(Xtr, ytr, Xtst, ytst, config.kernel_params, inducing_pts, config.lambd, config.task,
                    a0, config.max_iter, config.device, logger)
            else:
                opt.run(Xtr, ytr, Xtst, ytst, config.kernel_params, config.lambd, config.task,
                        a0, config.max_iter, config.device, logger)


if __name__ == '__main__':
    main()