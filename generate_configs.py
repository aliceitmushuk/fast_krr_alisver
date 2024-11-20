import os
import yaml
import itertools

from src.data_configs import DATA_KEYS, DATA_CONFIGS, KERNEL_CONFIGS, LAMBDA_CONFIGS

SEED = 0

PRECONDITIONERS = [None, "nystrom", "partial_cholesky"]
CHOLESKY_MODES = ["greedy", "rpc"]
BLOCK_SAMPLING = ["uniform", "rls"]
ACCELERATED = [True, False]


def generate_combinations(sweep_params):
    """
    Generate all combinations of sweep parameters, subject to certain constraints.
    :param sweep_params: Dictionary where keys are parameter names and values are lists
    of possible values.
    :return: List of dictionaries, each representing a unique combination of parameters.
    """
    keys, values = zip(*sweep_params.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    new_combinations = []
    for config in combinations:
        # Update the wandb project
        config["wandb.project"] = f"{config['wandb.project']}_{config['dataset']}"

        # Add the task
        config["task"] = DATA_CONFIGS[config["dataset"]]["task"]

        # Add the kernel parameters
        # config["kernel"] = KERNEL_CONFIGS[config["dataset"]]
        kernel_params = KERNEL_CONFIGS[config["dataset"]]
        config["kernel.type"] = kernel_params["type"]
        config["kernel.sigma"] = kernel_params["sigma"]
        if "nu" in kernel_params:
            config["kernel.nu"] = kernel_params["nu"]

        # Add the regularization parameter
        config["lambd_unscaled"] = LAMBDA_CONFIGS[config["dataset"]]

        # Add ASkotchV2 parameters
        if config["opt.type"] == "askotchv2":
            config["opt.block_sz_frac"] = 0.1  # converted downstream to block_sz
            config["opt.mu"] = None  # converted downstream to the actual value
            config["opt.nu"] = None  # converted downstream to the actual value
            # apply itertools.product to this config, PRECONDITIONERS, BLOCK_SAMPLING,
            # ACCELERATED
            for precond, block_sampling, accelerated in itertools.product(
                PRECONDITIONERS, BLOCK_SAMPLING, ACCELERATED
            ):
                if precond != "partial_cholesky":
                    new_config = config.copy()
                    new_config["precond.type"] = precond
                    new_config[
                        "precond.rho"
                    ] = None  # converted downstream to the actual value
                    new_config["opt.sampling_method"] = block_sampling
                    new_config["opt.accelerated"] = accelerated
                    new_combinations.append(new_config)
        elif config["opt.type"] == "pcg":
            # apply itertools.product to this config, PRECONDITIONERS, CHOLESKY_MODES
            # only do cholesky_mode if precond is "partial_cholesky"
            for precond in PRECONDITIONERS:
                if precond != "partial_cholesky":
                    new_config = config.copy()
                    new_config["precond.type"] = precond
                    new_config[
                        "precond.rho"
                    ] = None  # converted downstream to the actual value
                    new_combinations.append(new_config)
                else:
                    for cholesky_mode in CHOLESKY_MODES:
                        new_config = config.copy()
                        new_config["precond.type"] = precond
                        new_config[
                            "precond.rho"
                        ] = None  # converted downstream to the actual value
                        new_config["precond.mode"] = cholesky_mode
                        new_combinations.append(new_config)
    return new_combinations


def save_configs(combinations, output_dir):
    """
    Save each configuration as a YAML file in a structured folder hierarchy.
    :param combinations: List of dictionaries representing parameter combinations.
    :param output_dir: Root directory to save configurations.
    """
    for combo in combinations:
        # Generate folder path based on parameters
        folder_path = os.path.join(
            output_dir,
            "/".join(
                [
                    f"{key}_{value}"
                    for key, value in combo.items()
                    if key != "kernel.sigma"
                ]
            ),  # Avoid repeating derived values
        )
        os.makedirs(folder_path, exist_ok=True)

        # Save config.yaml
        config_path = os.path.join(folder_path, "config.yaml")
        with open(config_path, "w") as file:
            yaml.dump(combo, file, default_flow_style=False)

        print(f"Generated: {config_path}")


if __name__ == "__main__":
    # Define all possible values for each parameter
    # sweep_params = {
    #     "dataset": ["homo", "susy"],
    #     "model": ["full_krr", "inducing_krr"],
    #     "kernel.type": ["l1_laplace", "rbf"],
    #     "precond.type": ["nystrom", "chol"],
    #     "precond.rank": [10, 20],
    #     "training.max_time": [3600],
    #     "training.log_freq": [50],
    #     "training.precision": ["float32"],
    #     "training.seed": [0],
    #     "wandb.project": ["default_project"]
    # }

    datasets_to_remove = [
        "synthetic",
        "airlines",
        "acsincome",
        "click_prediction",
        "susy",
        "higgs",
        "taxi",
    ]
    datasets_performance = [
        dataset for dataset in DATA_KEYS if dataset not in datasets_to_remove
    ]

    # Common to all runs for full KRR performance
    sweep_params_performance_full_krr = {
        "dataset": datasets_performance,
        "model": ["full_krr"],
        "opt.type": ["askotchv2", "pcg"],
        "precond.r": [300],
        "training.max_time": [7200],
        "training.log_freq": [50],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.log_test_only": [False],
        "wandb.project": ["performance_full_krr"],
    }

    # Output directory for experiment configurations
    output_dir = "performance_full_krr"

    # Generate all parameter combinations
    combinations = generate_combinations(sweep_params_performance_full_krr)

    # Save configurations to folder structure
    save_configs(combinations, output_dir)
