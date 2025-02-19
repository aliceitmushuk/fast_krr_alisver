import itertools

from fast_krr.data_configs import DATA_CONFIGS, PERFORMANCE_DATASETS
from fast_krr.experiment_configs import (
    KERNEL_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
    EIGENPRO3_INDUCING_POINTS_GRID,
)
from fast_krr.generate_configs_utils import (
    add_kernel_params,
    get_nested_config,
    save_configs,
)


SEED = 0
BLOCKSZ = 12_000
R = 100


def generate_eigenpro3_configs(base_config):
    configs = []
    config = base_config.copy()
    config["opt"] = {
        "type": "eigenpro3",
        "block_sz": BLOCKSZ,
        "r": R,
        "bg": None,
        "proj_inner_iters": None,
    }
    configs.append(config)
    return configs


def generate_combinations(
    sweep_params,
    kernel_configs,
    data_configs,
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
        nested_config["lambd_unscaled"] = 0.0  # EigenPro3 does not use regularization
        all_combinations.extend(generate_eigenpro3_configs(nested_config))

    return all_combinations


if __name__ == "__main__":
    sweep_params_performance_eigenpro3 = {
        "dataset": PERFORMANCE_DATASETS,
        "model": ["inducing_krr"],
        "m": EIGENPRO3_INDUCING_POINTS_GRID,
        "opt.type": ["eigenpro3"],
        "training.log_freq": [20],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }

    output_dir = "performance_inducing_krr_ep3"

    combinations_eigenpro3 = generate_combinations(
        sweep_params_performance_eigenpro3,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    save_configs(combinations_eigenpro3, output_dir)
