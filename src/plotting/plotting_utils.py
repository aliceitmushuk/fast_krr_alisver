import os
import warnings

import numpy as np
import wandb
import matplotlib.pyplot as plt

from sorting import sort_data


# Not ideal - really should record this within runs
TRAINING_SIZE = {
    "synthetic": 10000,
    "homo": 100000,
    "susy": 4500000,
    "higgs": 10500000,
    "taxi_sub": 100000000,
}

LINESTYLES = {
    "askotch": "solid",
    "skotch": "dotted",
    "pcg": "dashed",
    "sketchysaga": "dashdot",
    "sketchykatyusha": "dashdotdotted",
}

RANK_COLORS = {
    10: "tab:blue",
    20: "tab:orange",
    50: "tab:green",
    100: "tab:red",
    200: "tab:purple",
    300: "tab:brown",
    500: "tab:pink",
    1000: "tab:gray",
    2000: "tab:olive",
}

BLOCK_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    5: "tab:green",
    10: "tab:red",
    20: "tab:purple",
    50: "tab:brown",
    100: "tab:pink",
    200: "tab:gray",
    500: "tab:olive",
    1000: "tab:cyan",
    2000: "gold",
}

# Useful for distinguishing between PCG methods
MARKERS_PRECOND = {
    "nystrom": "o",
    "partial_cholesky": "s",
    "falkon": None,
}

MARKEVERY = 1
MARKERSIZE = 5

METRIC_LABELS = {
    "rel_residual": "Relative residual",
    "train_loss": "Training loss",
    "test_acc": "Test accuracy",
    "test_mse": "Test MSE",
    "test_rmse": "Test RMSE",
    "smape": "Test SMAPE",
    "rel_suboptim": "Relative suboptimality",
}

OPT_LABELS = {
    "askotch": "ASkotch",
    "skotch": "Skotch",
    "pcg": "PCG",
    "sketchysaga": "SketchySAGA",
    "sketchykatyusha": "SketchyKatyusha",
}

HYPERPARAM_LABELS = {
    "r": "r",
    "b": "B",
    "precond": {"nystrom": r"Nystr$\ddot{\mathrm{o}}$m",
                "partial_cholesky": "Greedy Cholesky",
                "falkon": "Falkon",},
}

PCG_LABELS = {
    "nystrom": "Nystr$\ddot{\mathrm{o}}$mPCG",
    "partial_cholesky": "CholeskyPCG",
    "falkon": "Falkon",
}

X_AXIS_LABELS = {
    "time": "Time (s)",
    "datapasses": "Full data passes",
    "iters": "Iterations",
}

def set_fontsize(fontsize):
    plt.rcParams.update({"font.size": fontsize})

def render_in_latex():
    plt.rcParams.update({"text.usetex": True,
                          "font.family": "serif"})

def get_project_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return runs

def check_criteria(run, criteria):
    for _, criterion in criteria.items():
        if not criterion(run):
            return False
    return True

def filter_runs(runs, criteria):
    return [run for run in runs if check_criteria(run, criteria)]

def get_datapasses(run, steps):
    opt = run.config["opt"]
    n = TRAINING_SIZE[run.config["dataset"]]
    m = run.config["m"] if "m" in run.config else None
    bg = run.config["bg"] if "bg" in run.config else None
    p = run.config["p"] if "p" in run.config else None
    scaling_factor = None

    if opt in ["skotch", "askotch"]:
        scaling_factor = 1 / run.config["b"]
    elif opt == "pcg":
        if run.config["precond_params"]["type"] == "falkon":
            scaling_factor = (2 * m * n + m ** 2) / (n ** 2)
        else:
            scaling_factor = 1
    elif opt == "sketchysaga":
        scaling_factor = (2 * m * bg + m ** 2) / (n ** 2)
    elif opt == "sketchykatyusha":
        scaling_factor = (2 * m * bg + m ** 2 + p * 2 * m * n) / (n ** 2)

    return scaling_factor * steps

def get_x(run, steps, x_axis):
    if x_axis == "time":
        times_hist = run.scan_history(keys=["iter_time"])
        times = np.array([time["iter_time"] for time in times_hist])
        cum_times = np.cumsum(times)
        return cum_times[steps]
    elif x_axis == "datapasses":
        return get_datapasses(run, steps)
    elif x_axis == "iters":
        return steps

def get_label(run, hparams_to_label_opt):
    label = OPT_LABELS[run.config["opt"]]

    for hparam in hparams_to_label_opt:
        if hparam not in ["r", "b", "precond"]:
            raise ValueError(f"Unknown hparam: {hparam}")
        
        if hparam == "r" and run.config["precond_params"] is not None:
            label += f", ${HYPERPARAM_LABELS[hparam]} = {run.config['precond_params']['r']}$"
        elif hparam == "b" and "b" in run.config:
            label += f", ${HYPERPARAM_LABELS[hparam]} = {run.config['b']}$"
        elif hparam == "precond" and run.config["precond_params"] is not None:
            precond_type = run.config["precond_params"]["type"]
            if run.config["opt"] == "pcg":
                label = PCG_LABELS[precond_type]
                if precond_type == "falkon":
                    label += f", $m = {run.config['m']}$"
            else:
                label += f", {HYPERPARAM_LABELS[hparam][precond_type]}"

    return label

def get_style(run, color_param):
    style = {}
    opt = run.config["opt"]
    style["linestyle"] = LINESTYLES[opt]

    # Get color based on color_param
    if run.config["opt"] != "pcg" and color_param == "r" \
        and run.config["precond_params"] is not None \
            and "r" in run.config["precond_params"]:
        style["color"] = RANK_COLORS[run.config["precond_params"]["r"]]
    elif run.config["opt"] in ["skotch", "askotch"] and color_param == "b":
        style["color"] = BLOCK_COLORS[run.config["b"]]
    else:
        style["color"] = "k"

    # Use markers for PCG methods
    if opt == "pcg":
        style["marker"] = MARKERS_PRECOND[run.config["precond_params"]["type"]]
        style["markevery"] = MARKEVERY
        style["markersize"] = MARKERSIZE

    return style

def get_save_path(save_dir, save_name):
    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return os.path.join(save_dir, save_name)
    elif save_dir is not None and save_name is None:
        warnings.warn("Must provide save_name if save_dir is provided. Plot will not be saved.")
        return None
    elif save_dir is None and save_name is not None:
        warnings.warn("Must provide save_dir if save_name is provided. Plot will not be saved.")
        return None
    else:
        return None

def plot_runs(run_list, hparams_to_label, color_param, metric, x_axis, ylim, title,
               save_dir=None, save_name=None):
    if x_axis not in ["time", "datapasses", "iters"]:
        raise ValueError(f"Unsupported value of x_axis: {x_axis}")
    
    if color_param not in ["r", "b"]:
        raise ValueError(f"Unsupported value of color_param: {color_param}")
    
    # Sort the runs based on opt, color_param, and preconditioner type
    run_list = sort_data(run_list, sort_keys=["opt", color_param, "preconditioner_type"])
    
    save_path = get_save_path(save_dir, save_name)

    plt.figure()

    for run in run_list:
        y_hist = run.scan_history(keys=[metric, "_step"])
        y = np.array([hist[metric] for hist in y_hist])
        steps = np.array([hist["_step"] for hist in y_hist])

        x = get_x(run, steps, x_axis)
        label = get_label(run, hparams_to_label[run.config["opt"]])
        style = get_style(run, color_param)

        plt.plot(x, y, label=label, **style)

    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(X_AXIS_LABELS[x_axis])
    plt.ylabel(METRIC_LABELS[metric])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

def plot_runs_rel_suboptim(run_list, hparams_to_label, color_param, train_loss_optim, x_axis, ylim, title,
               save_dir=None, save_name=None):
    if x_axis not in ["time", "datapasses", "iters"]:
        raise ValueError(f"Unsupported value of x_axis: {x_axis}")
    
    if color_param not in ["r", "b"]:
        raise ValueError(f"Unsupported value of color_param: {color_param}")
    
    # Sort the runs based on opt, color_param, and preconditioner type
    run_list = sort_data(run_list, sort_keys=["opt", color_param, "preconditioner_type"])
    
    save_path = get_save_path(save_dir, save_name)

    plt.figure()

    for run in run_list:
        y_hist = run.scan_history(keys=["train_loss", "_step"])
        y = np.array([hist["train_loss"] for hist in y_hist])
        steps = np.array([hist["_step"] for hist in y_hist])

        x = get_x(run, steps, x_axis)
        label = get_label(run, hparams_to_label[run.config["opt"]])
        style = get_style(run, color_param)

        plt.semilogy(x, np.abs((y - train_loss_optim) / train_loss_optim), label=label, **style)

    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(X_AXIS_LABELS[x_axis])
    plt.ylabel(METRIC_LABELS["rel_suboptim"])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")