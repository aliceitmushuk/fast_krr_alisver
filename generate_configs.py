import os
import yaml
import itertools

from src.experiment_configs import (
    # DATA_KEYS,
    DATA_CONFIGS,
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    FULL_KRR_PERFORMANCE_TIME_CONFIGS,
)

SEED = 0

PRECONDITIONERS = [None, "nystrom", "partial_cholesky"]
CHOLESKY_MODES = ["greedy", "rpc"]
BLOCK_SAMPLING = ["uniform", "rls"]
ACCELERATED = [True, False]
PRECOND_RHO = ["damped", "regularization"]
BLK_SZ_FRAC = 0.1


def add_kernel_params(config):
    kernel_params = KERNEL_CONFIGS[config["dataset"]]
    config["kernel"] = {
        "type": kernel_params["type"],
        "sigma": kernel_params["sigma"],
    }
    if "nu" in kernel_params:
        config["kernel"]["nu"] = kernel_params["nu"]


def add_preconditioner_config(base_config, precond, additional_params=None):
    additional_params = additional_params or {}
    new_config = base_config.copy()
    new_config["precond"] = {"type": precond}

    if precond == "nystrom":
        for rho in PRECOND_RHO:
            rho_config = new_config.copy()
            rho_config["precond"]["rho"] = rho
            rho_config["precond"]["r"] = base_config.get("precond.r", None)
            yield {**rho_config, **additional_params}
    elif precond == "partial_cholesky":
        for cholesky_mode in CHOLESKY_MODES:
            cholesky_config = new_config.copy()
            cholesky_config["precond"]["mode"] = cholesky_mode
            cholesky_config["precond"]["rho"] = "regularization"
            cholesky_config["precond"]["r"] = base_config.get("precond.r", None)
            yield {**cholesky_config, **additional_params}
    else:  # precond is None
        new_config["precond"]["rho"] = None
        new_config["precond"]["r"] = None
        yield {**new_config, **additional_params}


def generate_askotchv2_configs(base_config):
    for precond, block_sampling, accelerated in itertools.product(
        [None, "nystrom"], BLOCK_SAMPLING, ACCELERATED
    ):
        new_config = base_config.copy()

        # Update opt while retaining existing keys (e.g., opt.type)
        opt_updates = {
            "sampling_method": block_sampling,
            "accelerated": accelerated,
            "block_sz_frac": BLK_SZ_FRAC,
            "mu": None,
            "nu": None,
        }
        new_config["opt"].update(opt_updates)

        yield from add_preconditioner_config(new_config, precond)


def generate_pcg_configs(base_config):
    for precond in PRECONDITIONERS:
        new_config = base_config.copy()

        # Retain existing keys in opt (e.g., opt.type) without overwriting
        opt_updates = {}  # No additional keys for PCG at this stage
        new_config["opt"].update(opt_updates)

        yield from add_preconditioner_config(new_config, precond)


def generate_combinations(sweep_params):
    """
    Generate all combinations of sweep parameters, subject to certain constraints.
    :param sweep_params: Dictionary where keys are parameter names and values are lists
    of possible values.
    :return: List of nested dictionaries, each representing a
    unique combination of parameters.
    """
    keys, values = zip(*sweep_params.items())
    base_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    all_combinations = []

    for base_config in base_combinations:
        # Filter out invalid combinations
        if (
            base_config["opt.type"] == "askotchv2"
            and base_config["training.precision"] == "float64"
        ):
            continue

        # Update nested fields for wandb, task, and training time
        nested_config = {
            "wandb": {
                "project": f"{base_config['wandb.project']}_{base_config['dataset']}"
            },
            "task": DATA_CONFIGS[base_config["dataset"]]["task"],
            "training": {
                "max_time": FULL_KRR_PERFORMANCE_TIME_CONFIGS[base_config["dataset"]],
                "max_iter": base_config["training.max_iter"],
                "log_freq": base_config["training.log_freq"],
                "precision": base_config["training.precision"],
                "seed": base_config["training.seed"],
                "log_test_only": base_config["training.log_test_only"],
            },
            "device": base_config["device"],
            "model": base_config["model"],
            "dataset": base_config["dataset"],
            "opt": {"type": base_config["opt.type"]},
        }

        # Add kernel and regularization parameters
        add_kernel_params(nested_config)
        nested_config["lambd_unscaled"] = LAMBDA_CONFIGS[base_config["dataset"]]

        # Generate configurations based on opt.type
        if base_config["opt.type"] == "askotchv2":
            all_combinations.extend(generate_askotchv2_configs(nested_config))
        elif base_config["opt.type"] == "pcg":
            all_combinations.extend(generate_pcg_configs(nested_config))

    return all_combinations


def save_configs(combinations, output_dir):
    """
    Save each configuration as a YAML file in a structured folder hierarchy.
    :param combinations: List of dictionaries representing parameter combinations.
    :param output_dir: Root directory to save configurations.
    """
    for idx, combo in enumerate(combinations):
        # Generate folder path based on dataset and model
        folder_path = os.path.join(
            output_dir, combo["dataset"], combo["model"], f"config_{idx}"
        )
        os.makedirs(folder_path, exist_ok=True)

        # Save config.yaml
        config_path = os.path.join(folder_path, "config.yaml")
        with open(config_path, "w") as file:
            yaml.dump(combo, file, default_flow_style=False)

        print(f"Generated: {config_path}")


if __name__ == "__main__":
    # Example datasets (adjust as needed)
    datasets_performance = ["ijcnn1"]

    # Common to all runs for full KRR performance
    sweep_params_performance_full_krr = {
        "dataset": datasets_performance,
        "model": ["full_krr"],
        "opt.type": ["askotchv2", "pcg"],
        "precond.r": [100],
        "training.log_freq": [50],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.log_test_only": [False],
        "training.max_iter": [None],
        "device": [0],
        "wandb.project": ["performance_full_krr"],
    }

    # Output directory for experiment configurations
    output_dir = "performance_full_krr"

    # Generate all parameter combinations
    combinations = generate_combinations(sweep_params_performance_full_krr)

    # Save configurations to folder structure
    save_configs(combinations, output_dir)
