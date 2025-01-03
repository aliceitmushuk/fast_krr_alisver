from functools import partial

from constants import EXTENSION, FONTSIZE, X_AXIS, HPARAMS_TO_LABEL
from constants import ENTITY_NAME, PROJECT_FULL_KRR, PROJECT_INDUCING_KRR
from constants import PERFORMANCE_DATASETS
from utils import set_fontsize, plot_runs_grid
from utils import render_in_latex  # noqa: F401
from cfg_utils import get_grid_shape, get_save_name, get_filtered_krr_runs

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


def plot_runs_dataset_grid(
    entity_name,
    full_krr_cfg,
    inducing_krr_cfg,
    datasets_cfg,
    hparams_to_label,
    x_axis,
    name_stem,
    save_dir,
    extension,
):
    run_lists = []
    metrics = []
    ylims = []
    titles = []

    n_rows, n_cols = get_grid_shape(datasets_cfg)
    save_name = get_save_name(name_stem, datasets_cfg, extension)

    for ds, config in datasets_cfg["datasets"].items():
        runs_full_krr = get_filtered_krr_runs(full_krr_cfg, ds, entity_name)
        runs_inducing_krr = get_filtered_krr_runs(inducing_krr_cfg, ds, entity_name)

        run_lists.append(runs_full_krr + runs_inducing_krr)
        metrics.append(config["metric"])
        ylims.append(config["ylim"])
        titles.append(ds)

    plot_runs_grid(
        run_lists,
        hparams_to_label,
        metrics,
        x_axis,
        ylims,
        titles,
        n_cols,
        n_rows,
        save_dir,
        save_name,
    )


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

    for dataset in PERFORMANCE_DATASETS:
        full_krr_cfg = full_krr_base_cfg.copy()
        full_krr_cfg["criteria_list"] = full_krr_criteria_float32
        inducing_krr_cfg = inducing_krr_base_cfg.copy()
        inducing_krr_cfg["criteria_list"] = inducing_krr_criteria_float32
        plot_fn(
            datasets_cfg=dataset,
            full_krr_cfg=full_krr_cfg,
            inducing_krr_cfg=inducing_krr_cfg,
            name_stem=NAME_STEM + "float32_",
        )

        full_krr_cfg = full_krr_base_cfg.copy()
        full_krr_cfg["criteria_list"] = full_krr_criteria_float64
        inducing_krr_cfg = inducing_krr_base_cfg.copy()
        inducing_krr_cfg["criteria_list"] = inducing_krr_criteria_float64
        plot_fn(
            datasets_cfg=dataset,
            full_krr_cfg=full_krr_cfg,
            inducing_krr_cfg=inducing_krr_cfg,
            name_stem=NAME_STEM + "float64_",
        )
