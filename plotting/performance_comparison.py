from functools import partial

from constants import EXTENSION, FONTSIZE, X_AXIS, HPARAMS_TO_LABEL
from constants import ENTITY_NAME, PROJECT_FULL_KRR, PROJECT_INDUCING_KRR
from constants import PERFORMANCE_DATASETS_CFG
from base_utils import set_fontsize
from base_utils import render_in_latex  # noqa: F401
from cfg_utils import plot_runs_dataset_grid

# high-level plotting parameters
SAVE_DIR = "./plots/performance_comparison"
NAME_STEM = "askotch_vs_pcg_"

# run filters
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
    "rho_damped": lambda run: run.config.get("precond_params", {}).get("rho", None)
    == "damped",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT32_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float32",
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT64_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float64",
    "finished": lambda run: run.state == "finished",
}


if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    # render_in_latex()

    plot_fn = partial(
        plot_runs_dataset_grid,
        entity_name=ENTITY_NAME,
        hparams_to_label=HPARAMS_TO_LABEL,
        x_axis=X_AXIS,
        save_dir=SAVE_DIR,
        extension=EXTENSION,
    )

    full_krr_base_cfg = {
        "proj_name": PROJECT_FULL_KRR,
        "criteria_list": [ASKOTCH_FILTER],
    }
    inducing_krr_base_cfg = {
        "proj_name": PROJECT_INDUCING_KRR,
        "criteria_list": [],
    }
    full_krr_criteria_float32 = full_krr_base_cfg["criteria_list"] + [
        PCG_FLOAT32_FILTER
    ]
    full_krr_criteria_float64 = full_krr_base_cfg["criteria_list"] + [
        PCG_FLOAT64_FILTER
    ]
    inducing_krr_criteria_float32 = inducing_krr_base_cfg["criteria_list"] + [
        PCG_FLOAT32_FILTER
    ]
    inducing_krr_criteria_float64 = inducing_krr_base_cfg["criteria_list"] + [
        PCG_FLOAT64_FILTER
    ]

    for datasets_cfg in PERFORMANCE_DATASETS_CFG:
        full_krr_cfg = full_krr_base_cfg.copy()
        full_krr_cfg["criteria_list"] = full_krr_criteria_float32
        inducing_krr_cfg = inducing_krr_base_cfg.copy()
        inducing_krr_cfg["criteria_list"] = inducing_krr_criteria_float32
        plot_fn(
            datasets_cfg=datasets_cfg,
            full_krr_cfg=full_krr_cfg,
            inducing_krr_cfg=inducing_krr_cfg,
            name_stem=NAME_STEM + "float32_",
        )

        full_krr_cfg = full_krr_base_cfg.copy()
        full_krr_cfg["criteria_list"] = full_krr_criteria_float64
        inducing_krr_cfg = inducing_krr_base_cfg.copy()
        inducing_krr_cfg["criteria_list"] = inducing_krr_criteria_float64
        plot_fn(
            datasets_cfg=datasets_cfg,
            full_krr_cfg=full_krr_cfg,
            inducing_krr_cfg=inducing_krr_cfg,
            name_stem=NAME_STEM + "float64_",
        )
