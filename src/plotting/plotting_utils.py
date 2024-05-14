import numpy as np
import wandb
import matplotlib.pyplot as plt

MAX_SAMPLES = 1000000000 # Hacky way to get everything from wandb

# Not ideal - really should record this within runs
TRAINING_SIZE = {
    "synthetic": 10000,
    "homo": 100000,
    "susy": 4500000,
    "higgs": 10000000,
    "taxi_sub": 100000000,
}

LINESTYLES = {
    "askotch": "solid",
    "skotch": "dotted",
    "pcg": "dashed",
    "sketchysaga": "dashdot",
    "sketchykatyusha": "dotted",
}

# Color is based on rank
COLORS = {
    10: "tab:blue",
    20: "tab:orange",
    50: "tab:green",
    100: "tab:red",
    200: "tab:purple",
    300: "tab:olive",
    500: "tab:brown",
    1000: "tab:pink",
    2000: "tab:gray",
}

# Marker is based on number of blocks
MARKERS_BCD = {
    1: "o",
    2: "s",
    5: "^",
    10: "v",
    20: "<",
    50: ">",
    100: "p",
    200: "P",
    500: "*",
    1000: "h",
    2000: "H",
}

# Useful for distinguishing between PCG methods
MARKERS_PRECOND = {
    "nystrom": "+",
    "partial_cholesky": "x",
    "falkon": "D",
}

MARKEVERY = 10

METRIC_LABELS = {
    "rel_residual": "Relative residual",
    "train_loss": "Training loss",
    "test_acc": "Test accuracy",
    "test_mse": "Test MSE",
    "test_rmse": "Test RMSE",
    "smape": "SMAPE",
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
    # "precond": {"nystrom": r"Nystr$\ddot{\mathrm{o}}$m",
    #             "partial_cholesky": "Partial Cholesky",
    #             "falkon": "Falkon",},
    "precond": {"nystrom": r"Nystrom",
                "partial_cholesky": "Partial Cholesky",
                "falkon": "Falkon", },
}

X_AXIS_LABELS = {
    "time": "Time (s)",
    "datapasses": "Full Data passes",
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
    datapasses = []
    opt = run.config["opt"]
    n = TRAINING_SIZE[run.config["dataset"]]
    m = run.config["m"] if "m" in run.config else None
    bg = run.config["bg"] if "bg" in run.config else None
    p = run.config["p"] if "p" in run.config else None
    scaling_factor = None

    # if opt in ["skotch", "askotch"]:
    #     return steps / run.config["b"]
    # elif opt == "pcg":
    #     if run.config["precond_params"]["type"] == "falkon":
    #         return steps * run.config["m"] / n
    #     else:
    #         return steps
    # elif opt == "sketchysaga":
    #     return steps * run.config["bg"] / n
    # elif opt == "sketchykatyusha":
    #     # Account for full gradient computations
    #     return steps * run.config["bg"] / n + steps * run.config["p"]

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

def get_label(run, hparams_to_label_opt):
    label = OPT_LABELS[run.config["opt"]]

    for hparam in hparams_to_label_opt:
        if hparam not in ["r", "b", "precond"]:
            raise ValueError(f"Unknown hparam: {hparam}")
        
        if hparam == "r":
            label += f", {HYPERPARAM_LABELS[hparam]}={run.config['precond_params']['r']}"
        elif hparam == "b":
            label += f", {HYPERPARAM_LABELS[hparam]}={run.config['b']}"
        elif hparam == "precond":
            precond_type = run.config["precond_params"]["type"]
            label += f", {HYPERPARAM_LABELS[hparam][precond_type]}"

    return label

def get_style(run, hparams_to_label_opt):
    style = {}
    opt = run.config["opt"]
    style["linestyle"] = LINESTYLES[opt]

    for hparam in hparams_to_label_opt:
        if hparam not in ["r", "b", "precond"]:
            raise ValueError(f"Unknown hparam: {hparam}")

        if hparam == "r":
            style["color"] = COLORS[run.config["precond_params"]["r"]]
        # Set the marker differently depending on the optimization method
        elif hparam == "b" and opt in ["skotch", "askotch"]:
            style["marker"] = MARKERS_BCD[run.config["b"]]
            style["markevery"] = MARKEVERY
        elif hparam == "precond" and opt == "pcg":
            style["marker"] = MARKERS_PRECOND[run.config["precond_params"]["type"]]
            style["markevery"] = MARKEVERY

    return style

def plot_runs(run_list, hparams_to_label, metric, x_axis, ylim, title):
    if x_axis not in ["time", "datapasses", "iters"]:
        raise ValueError(f"Unsupported value of x_axis: {x_axis}")

    for run in run_list:
        y_df = run.history(samples=MAX_SAMPLES, keys=[metric])
        steps = y_df["_step"].to_numpy()

        if x_axis == "time":
            times_df = run.history(samples=MAX_SAMPLES, keys=["iter_time"])
            cum_times = np.cumsum(times_df["iter_time"].to_numpy())
            x = cum_times[steps]
        elif x_axis == "datapasses":
            x = get_datapasses(run, steps)
        elif x_axis == "iters":
            x = steps

        label = get_label(run, hparams_to_label[run.config["opt"]])
        style = get_style(run, hparams_to_label[run.config["opt"]])

        plt.plot(x, y_df[metric], label=label, **style)

    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(X_AXIS_LABELS[x_axis])
    plt.ylabel(METRIC_LABELS[metric])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)