from functools import partial

from plotting_utils import (
    set_fontsize,
    get_project_runs,
    filter_runs,
    plot_runs_grid,
)
from plotting_utils import render_in_latex  # noqa: F401

# high-level plotting parameters
SAVE_DIR = "./plots/performance_comparison"
EXTENSION = "pdf"
NAME_STEM = "askotch_vs_pcg_"
FONTSIZE = 14
X_AXIS = "time"
HPARAMS_TO_LABEL = {
    "askotchv2": ["precond", "r", "sampling_method"],
    "pcg": ["precond", "r", "m"],
}

# wandb project names
ENTITY_NAME = "sketchy-opts"
PROJECT_FULL_KRR = "performance_full_krr_v2_"
PROJECT_INDUCING_KRR = "performance_inducing_krr_"

# run filters
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
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

# dataset-specific plotting parameters
VISION = {
    "datasets": {
        "cifar10": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "fashion_mnist": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "mnist": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "svhn": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
    },
    "grid": {"n_rows": 2, "n_cols": 2},
    "name_ext": "vision",
}
PARTICLE_PHYSICS = {
    "datasets": {
        "miniboone": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "susy": {
            "ylim": [0.6, 0.9],
            "metric": "test_acc",
        },
        "higgs": {
            "ylim": [0.5, 0.8],
            "metric": "test_acc",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 3},
    "name_ext": "particle_physics",
}
TABULAR_CLASSIFICATION = {
    "datasets": {
        "covtype_binary": {
            "ylim": [0.0, 1.0],
            "metric": "test_acc",
        },
        "comet_mc": {
            "ylim": [0.4, 1.0],
            "metric": "test_acc",
        },
        "click_prediction": {
            "ylim": [0.4, 0.9],
            "metric": "test_acc",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 3},
    "name_ext": "tabular_classification",
}
QM9 = {
    "datasets": {
        "qm9": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 1},
    "name_ext": "qm9",
}
MOLECULES_BIG = {
    "datasets": {
        "toluene": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "ethanol": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "benzene": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "malonaldehyde": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 2, "n_cols": 2},
    "name_ext": "molecules_big",
}
MOLECULES_SMALL = {
    "datasets": {
        "uracil": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "aspirin": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "salicylic": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "naphthalene": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 2, "n_cols": 2},
    "name_ext": "molecules_small",
}
TABULAR_REGRESSION = {
    "datasets": {
        "yearpredictionmsd": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "acsincome": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 2},
    "name_ext": "tabular_regression",
}
ALL_DATASETS = [
    VISION,
    PARTICLE_PHYSICS,
    TABULAR_CLASSIFICATION,
    QM9,
    MOLECULES_BIG,
    MOLECULES_SMALL,
    TABULAR_REGRESSION,
]


def plot_runs_dataset_grid(
    entity_name,
    project_name_full_krr,
    project_name_inducing_krr,
    datasets_cfg,
    askotch_criteria,
    pcg_criteria,
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

    n_rows = datasets_cfg["grid"]["n_rows"]
    n_cols = datasets_cfg["grid"]["n_cols"]
    save_name = name_stem + datasets_cfg["name_ext"] + "." + extension

    for ds, config in datasets_cfg["datasets"].items():
        project_name_full_krr_ds = project_name_full_krr + ds
        project_name_inducing_krr_ds = project_name_inducing_krr + ds

        runs_full_krr = get_project_runs(entity_name, project_name_full_krr_ds)
        runs_inducing_krr = get_project_runs(entity_name, project_name_inducing_krr_ds)
        askotch_runs = filter_runs(runs_full_krr, askotch_criteria)
        pcg_full_krr_runs = filter_runs(runs_full_krr, pcg_criteria)
        pcg_inducing_krr_runs = filter_runs(runs_inducing_krr, pcg_criteria)

        run_lists.append(askotch_runs + pcg_full_krr_runs + pcg_inducing_krr_runs)
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
        project_name_full_krr=PROJECT_FULL_KRR,
        project_name_inducing_krr=PROJECT_INDUCING_KRR,
        askotch_criteria=ASKOTCH_FILTER,
        hparams_to_label=HPARAMS_TO_LABEL,
        x_axis=X_AXIS,
        save_dir=SAVE_DIR,
        extension=EXTENSION,
    )

    for dataset in ALL_DATASETS:
        plot_fn(
            datasets_cfg=dataset,
            pcg_criteria=PCG_FLOAT32_FILTER,
            name_stem=NAME_STEM + "float32_",
        )
        plot_fn(
            datasets_cfg=dataset,
            pcg_criteria=PCG_FLOAT64_FILTER,
            name_stem=NAME_STEM + "float64_",
        )
