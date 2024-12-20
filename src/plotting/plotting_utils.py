import os
import warnings

import numpy as np
import wandb
import matplotlib.pyplot as plt

from sorting import sort_data

OPT_COLORS = {
    "askotchv2": "tab:blue",
    "skotchv2": "tab:orange",
    "pcg": "black",
    "mimosa": "tab:pink",
}

PRECOND_MARKERS = {
    "nystrom": {"damped": "o", "regularization": "x"},
    "partial_cholesky": {"greedy": "s", "rpc": "v"},
    "falkon": "D",
}

SAMPLING_LINESTYLES = {
    "uniform": "solid",
    "rls": "dashed",
}

MARKEVERY = 1
MARKERSIZE = 8

METRIC_LABELS = {
    "rel_residual": "Relative residual",
    "train_loss": "Training loss",
    "test_acc": "Test accuracy",
    "test_mse": "Test MSE",
    "test_rmse": "Test RMSE",
    "test_mae": "Test MAE",
    "test_smape": "Test SMAPE",
    "rel_suboptim": "Relative suboptimality",
}

METRIC_PLOT_FNS = {
    "rel_residual": plt.semilogy,
    "train_loss": plt.plot,
    "test_acc": plt.plot,
    "test_mse": plt.plot,
    "test_rmse": plt.plot,
    "test_mae": plt.plot,
    "test_smape": plt.semilogy,
    "rel_suboptim": plt.semilogy,
}

METRIC_AX_PLOT_FNS = {
    "rel_residual": "semilogy",
    "train_loss": "plot",
    "test_acc": "plot",
    "test_mse": "plot",
    "test_rmse": "plot",
    "test_mae": "plot",
    "test_smape": "semilogy",
    "rel_suboptim": "semilogy",
}

OPT_LABELS = {
    "askotchv2": "ASkotch",
    "skotchv2": "Skotch",
    "pcg": "PCG",
    "mimosa": "Mimosa",
}

RANK_LABEL = "r"

PRECOND_LABELS = {
    "nystrom": r"Nystr$\ddot{\mathrm{o}}$m",
    "partial_cholesky": "Partial Cholesky",
    "falkon": "Falkon",
}

MODE_LABELS = {
    "greedy": "greedy",
    "rpc": "RPC",
}

RHO_LABELS = {
    "damped": "damped",
    "regularization": "regularization",
}

SAMPLING_LABELS = {
    "uniform": "uniform",
    "rls": "RLS",
}

X_AXIS_LABELS = {
    "time": "Time (s)",
    "datapasses": "Full data passes",
    "iters": "Iterations",
}

SORT_KEYS = ["opt", "accelerated", "sampling_method", "precond_type", "r", "m"]


def set_fontsize(fontsize):
    plt.rcParams.update({"font.size": fontsize})


def render_in_latex():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


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
    n = run.config["n"]
    m = run.config.get("m", None)
    bg = run.config.get("bg", None)
    block_sz = run.config.get("block_sz", None)
    scaling_factor = None

    if opt == "askotchv2":
        scaling_factor = block_sz / n
    elif opt == "pcg":
        if run.config["precond_params"]["type"] == "falkon":
            scaling_factor = (2 * m * n + m**2) / (n**2)
        else:
            scaling_factor = 1
    elif opt == "mimosa":
        scaling_factor = 2 * m * bg / (n**2)

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


def _rank_label(run):
    if run.config["precond_params"] is not None:
        return f"{RANK_LABEL} = {run.config['precond_params']['r']}"
    return None


def _precond_label(run):
    if run.config["precond_params"] is not None:
        precond_type = run.config["precond_params"]["type"]
        label = PRECOND_LABELS[precond_type]
        if precond_type == "nystrom":
            label += f", {RHO_LABELS[run.config['precond_params']['rho']]}"
        if precond_type == "partial_cholesky":
            label += f", {MODE_LABELS[run.config['precond_params']['mode']]}"
        return label
    return None


def _sampling_label(run):
    if "sampling_method" in run.config:
        return SAMPLING_LABELS[run.config["sampling_method"]]
    return None


def _inducing_label(run):
    if "m" in run.config:
        return f"m = {run.config['m']}"
    return None


def _get_opt(run):
    if run.config["opt"] == "askotchv2" and not run.config["accelerated"]:
        return "skotchv2"
    return run.config["opt"]


LABEL_FNS = {
    "r": _rank_label,
    "precond": _precond_label,
    "sampling_method": _sampling_label,
    "m": _inducing_label,
}


def get_label(run, hparams_to_label):
    opt = _get_opt(run)
    hparam_labels = [OPT_LABELS[opt]]
    for hparam in hparams_to_label:
        hparam_label = LABEL_FNS[hparam](run)
        if hparam_label is not None:
            hparam_labels.append(hparam_label)
    return ", ".join(hparam_labels)


def get_style(run):
    style = {}
    opt = _get_opt(run)
    style["color"] = OPT_COLORS[opt]

    if opt in ["askotchv2", "skotchv2"]:
        style["linestyle"] = SAMPLING_LINESTYLES[run.config["sampling_method"]]
    if run.config["precond_params"] is not None:
        precond_type = run.config["precond_params"]["type"]
        if precond_type == "nystrom":
            style["marker"] = PRECOND_MARKERS[precond_type][
                run.config["precond_params"]["rho"]
            ]
        elif precond_type == "partial_cholesky":
            style["marker"] = PRECOND_MARKERS[precond_type][
                run.config["precond_params"]["mode"]
            ]
        else:
            style["marker"] = PRECOND_MARKERS[precond_type]

    if opt == "pcg":
        style["markevery"] = MARKEVERY
        style["markersize"] = MARKERSIZE
    else:
        style["markevery"] = 5 * MARKEVERY
        style["markersize"] = MARKERSIZE
    return style


def get_save_path(save_dir, save_name):
    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return os.path.join(save_dir, save_name)
    elif save_dir is not None and save_name is None:
        warnings.warn(
            "Must provide save_name if save_dir is provided. Plot will not be saved."
        )
        return None
    elif save_dir is None and save_name is not None:
        warnings.warn(
            "Must provide save_dir if save_name is provided. Plot will not be saved."
        )
        return None
    else:
        return None


def plot_runs(
    run_list,
    hparams_to_label,
    metric,
    x_axis,
    ylim,
    title,
    save_dir=None,
    save_name=None,
):
    if x_axis not in ["time", "datapasses", "iters"]:
        raise ValueError(f"Unsupported value of x_axis: {x_axis}")
    plot_fn = METRIC_PLOT_FNS[metric]
    save_path = get_save_path(save_dir, save_name)

    run_list = sort_data(run_list, sort_keys=SORT_KEYS)

    plt.figure()

    for run in run_list:
        y_hist = run.scan_history(keys=[metric, "_step"])
        y = np.array([hist[metric] for hist in y_hist])
        steps = np.array([hist["_step"] for hist in y_hist])

        x = get_x(run, steps, x_axis)
        label = get_label(run, hparams_to_label[run.config["opt"]])
        style = get_style(run)

        plot_fn(x, y, label=label, **style)

    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(X_AXIS_LABELS[x_axis])
    plt.ylabel(METRIC_LABELS[metric])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_runs_axis(
    ax,
    run_list,
    hparams_to_label,
    metric,
    x_axis,
    ylim,
    title,
):
    plot_fn = getattr(ax, METRIC_AX_PLOT_FNS[metric])
    run_list = sort_data(run_list, sort_keys=SORT_KEYS)

    for run in run_list:
        y_hist = run.scan_history(keys=[metric, "_step"])
        y = np.array([hist[metric] for hist in y_hist])
        steps = np.array([hist["_step"] for hist in y_hist])

        x = get_x(run, steps, x_axis)
        label = get_label(run, hparams_to_label[run.config["opt"]])
        style = get_style(run)

        # Call the axis-specific plot function
        plot_fn(x, y, label=label, **style)

    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(X_AXIS_LABELS[x_axis])
    ax.set_ylabel(METRIC_LABELS[metric])


def plot_runs_grid(
    run_lists,
    hparams_to_label,
    metrics,
    x_axis,
    ylims,
    titles,
    n_cols,
    n_rows,
    save_dir=None,
    save_name=None,
):
    if x_axis not in ["time", "datapasses", "iters"]:
        raise ValueError(f"Unsupported value of x_axis: {x_axis}")
    save_path = get_save_path(save_dir, save_name)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_rows, 8 * n_cols))
    axes = axes.flatten()

    for i, (run_list, metric, ylim, title) in enumerate(
        zip(run_lists, metrics, ylims, titles)
    ):
        plot_runs_axis(axes[i], run_list, hparams_to_label, metric, x_axis, ylim, title)

    # Collect all handles and labels from all axes
    all_handles = []
    all_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # Deduplicate legend elements
    unique_labels = {}
    for h, l in zip(all_handles, all_labels):
        if l not in unique_labels:
            unique_labels[l] = h  # Keep the first occurrence of each label

    # Set the global legend
    plt.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
