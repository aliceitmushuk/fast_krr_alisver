from functools import partial

from tqdm import tqdm

from constants import (
    USE_LATEX,
    FONTSIZE,
    HPARAMS_TO_LABEL,
    BASE_SAVE_DIR,
    EXTENSION,
)
from constants import ENTITY_NAME
from constants import LIN_CVG
from base_utils import set_fontsize, render_in_latex
from cfg_utils import get_save_dir, create_krr_config, plot_runs_dataset_grid

# wandb project names
PROJECT_FULL_KRR = "performance_full_krr_v2_"
PROJECT_INDUCING_KRR = "performance_inducing_krr_"

# use a different x-axis for linear convergence
X_AXIS = "iters"

# save directory
SAVE_DIR = "lin_cvg"

# filters for runs
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "precision": lambda run: run.config["precision"] == "float64",
    "finished": lambda run: run.state == "finished",
}
MIMOSA_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "mimosa",
    "rho": lambda run: run.config.get("precond_params", {}).get("rho", None) == 3e1,
    "precision": lambda run: run.config["precision"] == "float64",
    "finished": lambda run: run.state == "finished",
}


if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    if USE_LATEX:
        render_in_latex()

    plot_fn = partial(
        plot_runs_dataset_grid,
        entity_name=ENTITY_NAME,
        hparams_to_label=HPARAMS_TO_LABEL,
        x_axis=X_AXIS,
        save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
        extension=EXTENSION,
    )

    full_krr_cfg_float64 = create_krr_config(PROJECT_FULL_KRR, [ASKOTCH_FILTER])
    inducing_krr_cfg_float64 = create_krr_config(PROJECT_INDUCING_KRR, [MIMOSA_FILTER])

    with tqdm(total=2, desc="Linear convergence") as pbar:
        plot_fn(
            full_krr_cfg=full_krr_cfg_float64,
            inducing_krr_cfg=None,
            datasets_cfg=LIN_CVG,
            name_stem="full_",
        )
        pbar.update(1)
        plot_fn(
            full_krr_cfg=None,
            inducing_krr_cfg=inducing_krr_cfg_float64,
            datasets_cfg=LIN_CVG,
            name_stem="inducing_",
        )
        pbar.update(1)
