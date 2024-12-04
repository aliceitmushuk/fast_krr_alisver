import os
import yaml


def add_kernel_params(config, kernel_configs):
    kernel_params = kernel_configs[config["dataset"]]
    config["kernel"] = {
        "type": kernel_params["type"],
        "sigma": kernel_params["sigma"],
    }
    if "nu" in kernel_params:
        config["kernel"]["nu"] = kernel_params["nu"]


def generate_nystrom_configs(base_config, rho_modes, r, use_cpu=False):
    configs = []
    for rho in rho_modes:
        config = base_config.copy()
        config["precond"] = {
            "type": "nystrom",
            "rho": rho,
            "r": r,
            "use_cpu": use_cpu,
        }
        configs.append(config)
    return configs


def generate_partial_cholesky_configs(base_config, chol_modes, r):
    configs = []
    for cholesky_mode in chol_modes:
        config = base_config.copy()
        config["precond"] = {
            "type": "partial_cholesky",
            "mode": cholesky_mode,
            "rho": "regularization",
            "r": r,
        }
        configs.append(config)
    return configs


def generate_falkon_configs(base_config):
    config = base_config.copy()
    config["precond"] = {
        "type": "falkon",
    }
    return [config]


def generate_no_preconditioner_configs(base_config):
    config = base_config.copy()
    config["precond"] = {
        "type": None,
        "rho": None,
        "r": None,
    }
    return [config]


def get_nested_config(
    base_config, data_configs, performance_time_configs, log_test_only
):
    nested_config = {
        "wandb": {
            "project": f"{base_config['wandb.project']}_{base_config['dataset']}"
        },
        "task": data_configs[base_config["dataset"]]["task"],
        "training": {
            "max_time": performance_time_configs[base_config["dataset"]],
            "max_iter": base_config["training.max_iter"],
            "log_freq": base_config["training.log_freq"],
            "precision": base_config["training.precision"],
            "seed": base_config["training.seed"],
            "log_test_only": log_test_only[base_config["dataset"]],
        },
        "model": base_config["model"],
        "dataset": base_config["dataset"],
    }
    if "m" in base_config:
        nested_config["m"] = base_config["m"]
    if "precond.r" in base_config:
        nested_config["precond"] = {"r": base_config["precond.r"]}
    else:
        nested_config["precond"] = {}

    return nested_config


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
