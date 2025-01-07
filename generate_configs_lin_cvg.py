from src.data_configs import DATA_CONFIGS
from src.experiment_configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
)
from src.generate_configs_utils import save_configs
from generate_configs_full_krr import generate_combinations as gc_full_krr
from generate_configs_falkon import generate_combinations as gc_falkon
from generate_configs_mimosa import generate_combinations as gc_mimosa

SEED = 0

PRECONDITIONERS = ["nystrom"]
CHOLESKY_MODES = [None]
SAMPLING_MODES = ["uniform", "rls"]
ACC_MODES = [True]
BG_MODES = [256]
RHO_MODES = ["damped", "regularization"]
RHO_MODES_MIMOSA = [1e0, 3e0, 1e1, 3e1, 1e2]
BLK_SZ_FRAC = 0.1

if __name__ == "__main__":
    sweep_params_askotchv2 = {
        "dataset": ["synthetic"],
        "model": ["full_krr"],
        "opt.type": ["askotchv2"],
        "precond.r": [50, 100, 200, 500],
        "training.log_freq": [20],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr_v2"],
    }
    sweep_params_pcg = {
        "dataset": ["synthetic"],
        "model": ["full_krr"],
        "opt.type": ["pcg"],
        "precond.r": [500],
        "training.log_freq": [20],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr_v2"],
    }
    sweep_params_falkon = {
        "dataset": ["synthetic"],
        "model": ["inducing_krr"],
        "m": [1000],
        "opt.type": ["pcg"],
        "training.log_freq": [20],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }
    sweep_params_mimosa = {
        "dataset": ["synthetic"],
        "model": ["inducing_krr"],
        "m": [1000],
        "opt.type": ["mimosa"],
        "precond.r": [50],
        "precond.use_cpu": [False],
        "training.log_freq": [200],
        "training.precision": ["float64"],
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
    combinations_falkon = gc_falkon(
        sweep_params_falkon,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    combinations_mimosa = gc_mimosa(
        sweep_params_mimosa,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
        RHO_MODES_MIMOSA,
        BG_MODES,
    )

    save_configs(combinations_askotchv2 + combinations_pcg, "lin_cvg_full_krr")
    save_configs(combinations_falkon, "lin_cvg_falkon")
    save_configs(combinations_mimosa, "lin_cvg_mimosa")
