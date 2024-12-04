import itertools
from pprint import pprint

from src.data_configs import DATA_CONFIGS, FALKON_DATASETS
from src.experiment_configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
    FALKON_INDUCING_POINTS_GRID,
)
from src.generate_configs_utils import add_kernel_params, save_configs

SEED = 0


def generate_pcg_configs(base_config):
    configs = []
    config = base_config.copy()
    config["opt"] = {
        "type": "pcg",
    }
    config["precond"] = {
        "type": "falkon",
    }
    configs.append(config)
    return configs


def generate_combinations(sweep_params, kernel_configs):
    keys, values = zip(*sweep_params.items())
    base_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    all_combinations = []

    for base_config in base_combinations:
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
            "m": base_config["m"],
            "dataset": base_config["dataset"],
            "precond": {},
        }

        add_kernel_params(nested_config, kernel_configs)
        nested_config["lambd_unscaled"] = LAMBDA_CONFIGS[base_config["dataset"]]
        all_combinations.extend(generate_pcg_configs(nested_config))

    return all_combinations


if __name__ == "__main__":
    sweep_params_performance_falkon = {
        "dataset": FALKON_DATASETS,
        "model": ["inducing_krr"],
        "m": FALKON_INDUCING_POINTS_GRID,
        "opt.type": ["pcg"],
        "training.log_freq": [50],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_falkon"],
    }

    output_dir = "performance_falkon"

    combinations = generate_combinations(
        sweep_params_performance_falkon, KERNEL_CONFIGS
    )
    pprint(combinations[:5])  # Debug: Print a sample of generated combinations
    save_configs(combinations, output_dir)
