from data_handling.configs import DATA_CONFIGS
from experiment_handling.configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
    FALKON_INDUCING_POINTS_GRID,
)
from config_gen.utils import save_configs
from config_gen.full_krr import generate_combinations as gc_full_krr
from config_gen.eigenpro2 import generate_combinations as gc_eigenpro2
from config_gen.eigenpro3 import generate_combinations as gc_eigenpro3
from config_gen.falkon import generate_combinations as gc_falkon

SEED = 0

PRECONDITIONERS = ["nystrom"]
CHOLESKY_MODES = [None]
SAMPLING_MODES = ["uniform"]
ACC_MODES = [True]
RHO_MODES = ["damped", "regularization"]
BLK_SZ_FRAC = 0.0005

if __name__ == "__main__":
    sweep_params_askotchv2 = {
        "dataset": ["taxi"],
        "model": ["full_krr"],
        "opt.type": ["askotchv2"],
        "precond.r": [50, 100, 200, 500],
        "training.log_freq": [200],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }
    sweep_params_pcg = {
        "dataset": ["taxi"],
        "model": ["full_krr"],
        "opt.type": ["pcg"],
        "precond.r": [50],
        "training.log_freq": [200],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }
    sweep_params_eigenpro2 = {
        "dataset": ["taxi"],
        "model": ["full_krr"],
        "opt.type": ["eigenpro2"],
        "training.log_freq": [200],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }
    sweep_params_eigenpro3 = {
        "dataset": ["taxi"],
        "model": ["inducing_krr"],
        "m": [1_000_000],
        "opt.type": ["eigenpro3"],
        "training.log_freq": [50],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }
    sweep_params_falkon = {
        "dataset": ["taxi"],
        "model": ["inducing_krr"],
        "m": FALKON_INDUCING_POINTS_GRID,
        "opt.type": ["pcg"],
        "training.log_freq": [200],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }

    combinations_askotchv2 = gc_full_krr(
        sweep_params_askotchv2,
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
    combinations_pcg = gc_full_krr(
        sweep_params_pcg,
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
    combinations_eigenpro2 = gc_eigenpro2(
        sweep_params_eigenpro2,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    combinations_eigenpro3 = gc_eigenpro3(
        sweep_params_eigenpro3,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    combinations_falkon = gc_falkon(
        sweep_params_falkon,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )

    save_configs(combinations_askotchv2 + combinations_pcg, "taxi_full_krr")
    save_configs(combinations_eigenpro2, "taxi_eigenpro2")
    save_configs(combinations_eigenpro3, "taxi_eigenpro3")
    save_configs(combinations_falkon, "taxi_falkon")
