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
from src.generate_configs_utils import (
    add_kernel_params,
    generate_falkon_configs,
    get_nested_config,
    save_configs,
)

SEED = 0


def generate_pcg_configs(base_config):
    configs = []
    config = base_config.copy()
    config["opt"] = {
        "type": "pcg",
    }
    configs.extend(generate_falkon_configs(config))
    return configs


def generate_combinations(
    sweep_params,
    kernel_configs,
    data_configs,
    lambda_configs,
    performance_time_configs,
    log_test_only,
):
    keys, values = zip(*sweep_params.items())
    base_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    all_combinations = []

    for base_config in base_combinations:
        nested_config = get_nested_config(
            base_config, data_configs, performance_time_configs, log_test_only
        )
        add_kernel_params(nested_config, kernel_configs)
        nested_config["lambd_unscaled"] = lambda_configs[base_config["dataset"]]
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
        sweep_params_performance_falkon,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    pprint(combinations[:5])  # Debug: Print a sample of generated combinations
    save_configs(combinations, output_dir)
