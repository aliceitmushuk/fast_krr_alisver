import os
import yaml
import itertools

from src.data_configs import DATA_KEYS, DATA_CONFIGS, KERNEL_CONFIGS, LAMBDA_CONFIGS

SEED = 0

PRECONDITIONERS = [None, "nystrom", "partial_cholesky"]
CHOLESKY_MODES = ["greedy", "rpc"]
BLOCK_SAMPLING = ["uniform", "rls"]
ACCELERATED = [True, False]
PRECOND_RHO = ["damped", "regularization"]
BLK_SZ_FRAC = 0.1


def add_kernel_params(config):
    kernel_params = KERNEL_CONFIGS[config["dataset"]]
    config["kernel.type"] = kernel_params["type"]
    config["kernel.sigma"] = kernel_params["sigma"]
    if "nu" in kernel_params:
        config["kernel.nu"] = kernel_params["nu"]


def add_preconditioner_config(base_config, precond, additional_params=None):
    additional_params = additional_params or {}
    new_config = base_config.copy()
    new_config["precond.type"] = precond
    if precond == "nystrom":
        for rho in PRECOND_RHO:
            rho_config = new_config.copy()
            rho_config["precond.rho"] = rho
            yield {**rho_config, **additional_params}
    else:
        new_config["precond.rho"] = "regularization"
        yield {**new_config, **additional_params}


def generate_askotchv2_configs(base_config):
    for precond, block_sampling, accelerated in itertools.product(
        PRECONDITIONERS, BLOCK_SAMPLING, ACCELERATED
    ):
        sampling_acc_params = {
            "opt.sampling_method": block_sampling,
            "opt.accelerated": accelerated,
        }
        yield from add_preconditioner_config(base_config, precond, sampling_acc_params)


def generate_pcg_configs(base_config):
    for precond in PRECONDITIONERS:
        if precond == "partial_cholesky":
            for cholesky_mode in CHOLESKY_MODES:
                yield from add_preconditioner_config(
                    base_config, precond, {"precond.mode": cholesky_mode}
                )
        else:
            yield from add_preconditioner_config(base_config, precond)


def generate_combinations(sweep_params):
    """
    Generate all combinations of sweep parameters, subject to certain constraints.
    :param sweep_params: Dictionary where keys are parameter names and values are lists
    of possible values.
    :return: List of dictionaries, each representing a unique combination of parameters.
    """
    keys, values = zip(*sweep_params.items())
    base_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    all_combinations = []

    for base_config in base_combinations:
        # Update the wandb project and task
        base_config[
            "wandb.project"
        ] = f"{base_config['wandb.project']}_{base_config['dataset']}"
        base_config["task"] = DATA_CONFIGS[base_config["dataset"]]["task"]

        # Add kernel and regularization parameters
        add_kernel_params(base_config)
        base_config["lambd_unscaled"] = LAMBDA_CONFIGS[base_config["dataset"]]

        # Generate configurations based on opt.type
        if base_config["opt.type"] == "askotchv2":
            if base_config["training.precision"] == "float64":
                continue  # Skip float64 for ASkotchV2
            base_config["opt.block_sz_frac"] = BLK_SZ_FRAC
            base_config["opt.mu"] = None
            base_config["opt.nu"] = None
            all_combinations.extend(generate_askotchv2_configs(base_config))
        elif base_config["opt.type"] == "pcg":
            all_combinations.extend(generate_pcg_configs(base_config))

    return all_combinations


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
        "precond.r": [100],
        "training.max_time": [7200],
        "training.log_freq": [50],
        "training.precision": ["float32", "float64"],
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
