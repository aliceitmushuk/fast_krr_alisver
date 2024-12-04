import os
import yaml
import itertools
from pprint import pprint
import glob

from src.data_configs import DATA_CONFIGS
from src.experiment_configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
)

SEED = 0

PRECONDITIONERS = ["nystrom", "partial_cholesky"]
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


def generate_nystrom_configs(base_config):
    configs = []
    for rho in PRECOND_RHO:
        config = base_config.copy()
        config["precond"] = {
            "type": "nystrom",
            "rho": rho,
            "r": base_config["precond"]["r"],
        }
        configs.append(config)
    return configs


def generate_partial_cholesky_configs(base_config):
    configs = []
    for cholesky_mode in CHOLESKY_MODES:
        config = base_config.copy()
        config["precond"] = {
            "type": "partial_cholesky",
            "mode": cholesky_mode,
            "rho": "regularization",
            "r": base_config["precond"]["r"],
        }
        configs.append(config)
    return configs


def generate_no_preconditioner_configs(base_config):
    config = base_config.copy()
    config["precond"] = {
        "type": None,
        "rho": None,
        "r": None,
    }
    return [config]


def generate_askotchv2_configs(base_config):
    configs = []
    for block_sampling, accelerated in itertools.product(BLOCK_SAMPLING, ACCELERATED):
        config = base_config.copy()
        config["opt"] = {
            "type": "askotchv2",
            "sampling_method": block_sampling,
            "accelerated": accelerated,
            "block_sz_frac": BLK_SZ_FRAC,
            "mu": None,
            "nu": None,
        }
        # Add all preconditioner configurations
        configs.extend(generate_nystrom_configs(config))
        configs.extend(generate_no_preconditioner_configs(config))
    return configs


def generate_pcg_configs(base_config):
    configs = []
    for precond in PRECONDITIONERS:
        config = base_config.copy()
        config["opt"] = {"type": "pcg"}
        if precond == "nystrom":
            configs.extend(generate_nystrom_configs(config))
        elif precond == "partial_cholesky":
            configs.extend(generate_partial_cholesky_configs(config))
    return configs


def generate_combinations(sweep_params):
    keys, values = zip(*sweep_params.items())
    base_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    all_combinations = []

    for base_config in base_combinations:
        if (
            base_config["opt.type"] == "askotchv2"
            and base_config["training.precision"] == "float64"
        ):
            continue

        nested_config = {
            "wandb": {
                "project": f"{base_config['wandb.project']}_{base_config['dataset']}"
            },
            "task": DATA_CONFIGS[base_config["dataset"]]["task"],
            "training": {
                "max_time": PERFORMANCE_TIME_CONFIGS[base_config["dataset"]],
                "max_iter": base_config["training.max_iter"],
                "log_freq": base_config["training.log_freq"],
                "precision": base_config["training.precision"],
                "seed": base_config["training.seed"],
                "log_test_only": LOG_TEST_ONLY[base_config["dataset"]],
            },
            "model": base_config["model"],
            "dataset": base_config["dataset"],
            "precond": {"r": base_config["precond.r"]},
        }

        add_kernel_params(nested_config)
        nested_config["lambd_unscaled"] = LAMBDA_CONFIGS[base_config["dataset"]]

        if base_config["opt.type"] == "askotchv2":
            all_combinations.extend(generate_askotchv2_configs(nested_config))
        elif base_config["opt.type"] == "pcg":
            all_combinations.extend(generate_pcg_configs(nested_config))

    return all_combinations


def save_configs(combinations, output_dir):
    for idx, combo in enumerate(combinations):
        folder_path = os.path.join(
            output_dir, combo["dataset"], combo["model"], f"config_{idx}"
        )
        os.makedirs(folder_path, exist_ok=True)

        config_path = os.path.join(folder_path, "config.yaml")
        with open(config_path, "w") as file:
            yaml.dump(combo, file, default_flow_style=False)

        print(f"Generated: {config_path}")


def validate_yaml_variations(output_dir):
    unique_values = {
        "CHOLESKY_MODES": set(),
        "BLOCK_SAMPLING": set(),
        "ACCELERATED": set(),
        "PRECOND_RHO": set(),
    }

    for file_path in glob.glob(f"{output_dir}/**/*.yaml", recursive=True):
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            unique_values["CHOLESKY_MODES"].add(config.get("precond", {}).get("mode"))
            unique_values["BLOCK_SAMPLING"].add(
                config.get("opt", {}).get("sampling_method")
            )
            unique_values["ACCELERATED"].add(config.get("opt", {}).get("accelerated"))
            unique_values["PRECOND_RHO"].add(config.get("precond", {}).get("rho"))

    for key, values in unique_values.items():
        print(f"{key}: {values}")
        # if None in values:
        #     print(f"Error: {key} contains unexpected None values!")


if __name__ == "__main__":
    datasets_performance = [
        dataset for dataset in DATA_CONFIGS.keys() if dataset != "taxi"
    ]

    sweep_params_performance_full_krr = {
        "dataset": datasets_performance,
        "model": ["full_krr"],
        "opt.type": ["askotchv2", "pcg"],
        "precond.r": [100],
        "training.log_freq": [50],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }

    output_dir = "performance_full_krr"

    combinations = generate_combinations(sweep_params_performance_full_krr)
    pprint(combinations[:5])  # Debug: Print a sample of generated combinations
    save_configs(combinations, output_dir)
    validate_yaml_variations(output_dir)
