import wandb
import argparse
import torch

from src.opts import bcd, abcd
from src.utils import set_random_seed, load_data

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='susy', help='Which dataset to use')
    parser.add_argument('--data_loc', type=str, default='./data/SUSY', help='Location of the dataset')
    parser.add_argument('--sigma', type=float, default=1.0, help='Kernel bandwidth')
    parser.add_argument('--lambd', type=float, default=0.1, help='Regularization parameter')
    parser.add_argument('--opt', choices=['bcd', 'abcd'], help='Which optimizer to use')
    parser.add_argument('--b', type=int, default=100, help='Number of blocks in optimizer')
    parser.add_argument('--r', type=int, default=10, help='Rank parameter in optimizer')
    parser.add_argument('--max_iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    parser.add_argument('--wandb_project', type=str, default='fast_krr', help='W&B project name')

    # Extract arguments from parser
    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    set_random_seed(seed)

    # Organize arguments for the experiment into a dictionary for logging in wandb
    experiment_args = {
        'dataset': args.dataset,
        'data_loc': args.data_loc,
        'sigma': args.sigma,
        'lambd': args.lambd,
        'opt': args.opt,
        'b': args.b,
        'r': args.r,
        'max_iter': args.max_iter,
        'seed': seed,
        'device': f'cuda:{args.device}'
    }

    # Print the experiment arguments
    print(f'Dataset: {experiment_args["dataset"]}')
    print(f'Data Location: {experiment_args["data_loc"]}')
    print(f'Sigma: {experiment_args["sigma"]}')
    print(f'Lambda: {experiment_args["lambd"]}')
    print(f'Optimizer: {experiment_args["opt"]}')
    print(f'# of Blocks: {experiment_args["b"]}')
    print(f'Rank: {experiment_args["r"]}')
    print(f'Max Iterations: {experiment_args["max_iter"]}')
    print(f'Seed: {experiment_args["seed"]}')
    print(f'Device: {experiment_args["device"]}')
    print(f'W&B Project: {args.wandb_project}')

    with wandb.init(project=args.wandb_project, config=experiment_args):
        # Access the experiment configuration
        config = wandb.config

        # Load the dataset
        Xtr, Xtst, ytr, ytst = load_data(config.dataset, config.data_loc, config.device)

        # Select the optimizer
        opt = bcd if config.opt == 'bcd' else abcd

        # Initialize at 0
        a0 = torch.zeros(Xtr.shape[0], device=config.device)

        # Run the optimizer
        with torch.no_grad():
            opt(Xtr, ytr, config.sigma, config.lambd, Xtst, ytst, a0, config.b, config.r, config.max_iter, config.device)

if __name__ == '__main__':
    main()