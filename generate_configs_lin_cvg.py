from src.data_configs import DATA_CONFIGS, SYNTHETIC_NTR
from src.experiment_configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    LOG_TEST_ONLY,
)
from src.generate_configs_utils import save_configs
from generate_configs_full_krr import generate_combinations as gc_full_krr
from generate_configs_mimosa import generate_combinations as gc_mimosa

SEED = 0

PRECONDITIONERS = ["nystrom"]
CHOLESKY_MODES = [None]
SAMPLING_MODES = ["uniform"]
ACC_MODES = [True]
BG_MODES = [256]
RHO_MODES = ["damped"]
RHO_MODES_MIMOSA = [1e0, 3e0, 1e1, 3e1, 1e2]
BLK_SZ_FRAC = 0.01
FAKE_TIME_CONFIG = {
    "comet_mc": 180000,
    "click_prediction": 180000,
    "acsincome": 180000,
    "synthetic": 180000,
}  # HACK(pratik): make time limit so large that max iterations is stopping criterion


def _get_askotch_combos(dataset, max_iter, blk_sz_frac):
    sweep_params_askotchv2_base = {
        "dataset": dataset,
        "model": ["full_krr"],
        "opt.type": ["askotchv2"],
        "training.log_freq": [100],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.max_iter": max_iter,
        "wandb.project": ["lin_cvg_full_krr"],
    }
    sweep_params_askotchv2 = {
        **sweep_params_askotchv2_base,
        "precond.r": [50, 100, 200, 500],
    }
    combinations_askotchv2 = gc_full_krr(
        sweep_params_askotchv2,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        RHO_MODES,
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        blk_sz_frac,
        PRECONDITIONERS,
    )
    return combinations_askotchv2, sweep_params_askotchv2_base


def _get_sap_combos_synthetic(sweep_params_askotchv2_base):
    sweep_params_sap50 = {
        **sweep_params_askotchv2_base,
        "precond.r": [50],
    }
    sweep_params_sap100 = {
        **sweep_params_askotchv2_base,
        "precond.r": [100],
    }
    sweep_params_sap200 = {
        **sweep_params_askotchv2_base,
        "precond.r": [200],
    }
    sweep_params_sap500 = {
        **sweep_params_askotchv2_base,
        "precond.r": [500],
    }

    # These should really be simplified using a partial function
    # Hardcoding with 50, 100, 200, 500 is also bad
    combinations_sap50 = gc_full_krr(
        sweep_params_sap50,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        ["regularization"],
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        50 / SYNTHETIC_NTR,
        ["newton"],
    )
    combinations_sap100 = gc_full_krr(
        sweep_params_sap100,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        ["regularization"],
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        100 / SYNTHETIC_NTR,
        ["newton"],
    )
    combinations_sap200 = gc_full_krr(
        sweep_params_sap200,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        ["regularization"],
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        200 / SYNTHETIC_NTR,
        ["newton"],
    )
    combinations_sap500 = gc_full_krr(
        sweep_params_sap500,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        ["regularization"],
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        500 / SYNTHETIC_NTR,
        ["newton"],
    )
    return (
        combinations_sap50
        + combinations_sap100
        + combinations_sap200
        + combinations_sap500
    )


def _get_mimosa_combos(dataset, max_iter):
    sweep_params_mimosa_base = {
        "dataset": dataset,
        "model": ["inducing_krr"],
        "m": [10_000],
        "opt.type": ["mimosa"],
        "precond.use_cpu": [False],
        "training.log_freq": [200],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.max_iter": max_iter,
        "wandb.project": ["lin_cvg_inducing_krr"],
    }
    sweep_params_mimosa = {
        **sweep_params_mimosa_base,
        "precond.r": [10, 20, 50, 100],
    }
    sweep_params_saga = {
        **sweep_params_mimosa_base,
        "precond.r": [None],
    }

    combinations_mimosa = gc_mimosa(
        sweep_params_mimosa,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        RHO_MODES_MIMOSA,
        BG_MODES,
    )
    combinations_saga = gc_mimosa(
        sweep_params_saga,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        FAKE_TIME_CONFIG,
        LOG_TEST_ONLY,
        RHO_MODES_MIMOSA,
        BG_MODES,
        [None],
    )
    return combinations_mimosa + combinations_saga


if __name__ == "__main__":
    datasets = ["comet_mc", "click_prediction", "acsincome", "uracil"]

    # Synthetic (ASkotch + SAP)
    combinations_askotchv2, sweep_params_askotchv2_base = _get_askotch_combos(
        ["synthetic"], [20_000], 0.1
    )
    combinations_sap = _get_sap_combos_synthetic(sweep_params_askotchv2_base)
    combinations_synthetic = combinations_askotchv2 + combinations_sap

    # All other datasets (ASkotch + Mimosa)
    combinations_datasets_askotchv2, _ = _get_askotch_combos(
        datasets, [20_000], BLK_SZ_FRAC
    )
    combinations_datasets_mimosa = _get_mimosa_combos(datasets, [100_000])

    save_configs(
        combinations_synthetic + combinations_datasets_askotchv2,
        "lin_cvg_full_krr",
    )
    save_configs(combinations_datasets_mimosa, "lin_cvg_inducing_krr")
