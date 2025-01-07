import itertools
import glob
import yaml

from src.data_configs import DATA_CONFIGS
from src.experiment_configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
)
from src.generate_configs_utils import (
    add_kernel_params,
    generate_nystrom_configs,
    generate_partial_cholesky_configs,
    generate_no_preconditioner_configs,
    get_nested_config,
    save_configs,
)


SEED = 0

PRECONDITIONERS = ["nystrom", "partial_cholesky"]
CHOLESKY_MODES = ["greedy", "rpc"]
SAMPLING_MODES = ["uniform", "rls"]
ACC_MODES = [True, False]
RHO_MODES = ["damped", "regularization"]
BLK_SZ_FRAC = 0.1


def generate_askotchv2_configs(
    base_config, rho_modes, sampling_modes, acc_modes, blk_sz_frac
):
    configs = []
    for block_sampling, accelerated in itertools.product(sampling_modes, acc_modes):
        config = base_config.copy()
        config["opt"] = {
            "type": "askotchv2",
            "sampling_method": block_sampling,
            "accelerated": accelerated,
            "block_sz_frac": blk_sz_frac,
            "mu": None,
            "nu": None,
        }
        # Add all preconditioner configurations
        configs.extend(
            generate_nystrom_configs(config, rho_modes, config["precond"]["r"])
        )
        configs.extend(generate_no_preconditioner_configs(config))
    return configs


def generate_pcg_configs(base_config, rho_modes, chol_modes, preconds):
    configs = []
    for precond in preconds:
        config = base_config.copy()
        config["opt"] = {"type": "pcg"}
        if precond == "nystrom":
            configs.extend(
                generate_nystrom_configs(config, rho_modes, config["precond"]["r"])
            )
        elif precond == "partial_cholesky":
            configs.extend(
                generate_partial_cholesky_configs(
                    config, chol_modes, config["precond"]["r"]
                )
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
    chol_modes,
    sampling_modes,
    acc_modes,
    blk_sz_frac,
    preconds,
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

        if base_config["opt.type"] == "askotchv2":
            all_combinations.extend(
                generate_askotchv2_configs(
                    nested_config, rho_modes, sampling_modes, acc_modes, blk_sz_frac
                )
            )
        elif base_config["opt.type"] == "pcg":
            all_combinations.extend(
                generate_pcg_configs(nested_config, rho_modes, chol_modes, preconds)
            )

    return all_combinations


def validate_yaml_variations(output_dir):
    unique_values = {
        "CHOLESKY_MODES": set(),
        "SAMPLING_MODES": set(),
        "ACC_MODES": set(),
        "RHO_MODES": set(),
    }

    for file_path in glob.glob(f"{output_dir}/**/*.yaml", recursive=True):
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            unique_values["CHOLESKY_MODES"].add(config.get("precond", {}).get("mode"))
            unique_values["SAMPLING_MODES"].add(
                config.get("opt", {}).get("sampling_method")
            )
            unique_values["ACC_MODES"].add(config.get("opt", {}).get("accelerated"))
            unique_values["RHO_MODES"].add(config.get("precond", {}).get("rho"))

    for key, values in unique_values.items():
        print(f"{key}: {values}")
        # if None in values:
        #     print(f"Error: {key} contains unexpected None values!")


if __name__ == "__main__":
    datasets_classification = [
        "mnist",
        "fashion_mnist",
        "cifar10",
        "svhn",
        "miniboone",
        "susy",
        "higgs",
        "covtype_binary",
        "comet_mc",
        "click_prediction",
    ]
    datasets_regression = [
        "qm9",
        "uracil",
        "aspirin",
        "salicylic",
        "naphthalene",
        "toluene",
        "ethanol",
        "benzene",
        "malonaldehyde",
        "yearpredictionmsd",
        "acsincome",
        "yolanda",
    ]
    datasets_performance = datasets_classification + datasets_regression

    sweep_params_performance_full_krr_askotchv2 = {
        "dataset": datasets_performance,
        "model": ["full_krr"],
        "opt.type": ["askotchv2"],
        "precond.r": [100],
        "training.log_freq": [20],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr_v2"],
    }

    sweep_params_performance_full_krr_pcg = {
        "dataset": datasets_performance,
        "model": ["full_krr"],
        "opt.type": ["pcg"],
        "precond.r": [100],
        "training.log_freq": [20],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr_v2"],
    }

    output_dir = "performance_full_krr_v2"

    combinations_askotchv2 = generate_combinations(
        sweep_params_performance_full_krr_askotchv2,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
        RHO_MODES,
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        BLK_SZ_FRAC,
        PRECONDITIONERS,
    )
    combinations_pcg = generate_combinations(
        sweep_params_performance_full_krr_pcg,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
        RHO_MODES,
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        BLK_SZ_FRAC,
        PRECONDITIONERS,
    )
    save_configs(combinations_askotchv2 + combinations_pcg, output_dir)
    validate_yaml_variations(output_dir)
