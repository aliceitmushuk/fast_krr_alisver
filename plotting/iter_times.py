from tqdm import tqdm

from constants import (
    USE_LATEX,
    FONTSIZE,
    BASE_SAVE_DIR,
    EXTENSION,
)

from constants import ENTITY_NAME, PROJECT_FULL_KRR
from constants import PERFORMANCE_DATASETS_CFG
from base_utils import set_fontsize, render_in_latex, plot_per_iter_runtime_ratios
from cfg_utils import get_save_dir, create_krr_config, _get_filtered_runs

# save directory
SAVE_DIR = "iter_times"

X_LABEL = "Per-iteration speedup"

# filters for runs
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
    "rho_damped": lambda run: run.config.get("precond_params", {}).get("rho", None)
    == "damped",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "block_sz_frac": lambda run: run.config["block_sz_frac"] == 0.01,
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT32_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float32",
    "not_greedy_cholesky": lambda run: not (
        run.config["precond_params"]["type"] == "partial_cholesky"
        and run.config["precond_params"]["mode"] == "greedy"
    ),
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT64_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float64",
    "not_greedy_cholesky": lambda run: not (
        run.config["precond_params"]["type"] == "partial_cholesky"
        and run.config["precond_params"]["mode"] == "greedy"
    ),
    "finished": lambda run: run.state == "finished",
}

if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    if USE_LATEX:
        render_in_latex()

    run_pairs_float32 = []
    run_pairs_float64 = []

    askotch_cfg_float32 = create_krr_config(PROJECT_FULL_KRR, [ASKOTCH_FILTER])
    pcg_cfg_float32 = create_krr_config(PROJECT_FULL_KRR, [PCG_FLOAT32_FILTER])
    pcg_cfg_float64 = create_krr_config(PROJECT_FULL_KRR, [PCG_FLOAT64_FILTER])

    # Loop through all configs in the datasets config and extract the dataset name
    for ds_config in PERFORMANCE_DATASETS_CFG:
        for ds in ds_config["datasets"].keys():
            runs_pcg_float32 = _get_filtered_runs(pcg_cfg_float32, ds, ENTITY_NAME)
            runs_pcg_float64 = _get_filtered_runs(pcg_cfg_float64, ds, ENTITY_NAME)
            runs_askotch = _get_filtered_runs(askotch_cfg_float32, ds, ENTITY_NAME)
            run_pairs_float32.append({"x": runs_pcg_float32, "y": runs_askotch})
            run_pairs_float64.append({"x": runs_pcg_float64, "y": runs_askotch})

    with tqdm(total=2, desc="Iteration times") as pbar:
        plot_per_iter_runtime_ratios(
            run_pairs_float32,
            x_label=X_LABEL,
            save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
            save_name=f"float32.{EXTENSION}",
        )
        pbar.update(1)
        plot_per_iter_runtime_ratios(
            run_pairs_float64,
            x_label=X_LABEL,
            save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
            save_name=f"float64.{EXTENSION}",
        )
        pbar.update(1)
