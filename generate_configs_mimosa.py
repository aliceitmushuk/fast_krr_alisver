import itertools

from src.generate_configs_utils import (
    add_kernel_params,
    get_nested_config,
    generate_nystrom_configs,
)


def generate_mimosa_configs(base_config, rho_modes, bg_modes):
    configs = []
    for bg in bg_modes:
        config = base_config.copy()
        config["opt"] = {
            "type": "mimosa",
            "bg": bg,
            "bH": None,
            "bH2": None,
        }
        # Add all preconditioner configurations
        configs.extend(
            generate_nystrom_configs(
                config, rho_modes, config["precond"]["r"], config["precond"]["use_cpu"]
            ),
        )
    return configs


def generate_combinations(
    sweep_params,
    kernel_configs,
    data_configs,
    lambda_configs,
    performance_time_configs,
    log_test_only,
    rho_modes,
    bg_modes,
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
        all_combinations.extend(
            generate_mimosa_configs(nested_config, rho_modes, bg_modes)
        )

    return all_combinations
