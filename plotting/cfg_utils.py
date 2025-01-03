from base_utils import get_project_runs, filter_runs_union, plot_runs_grid


def _get_grid_shape(datasets_cfg):
    n_rows = datasets_cfg["grid"]["n_rows"]
    n_cols = datasets_cfg["grid"]["n_cols"]
    return n_rows, n_cols


def _get_save_name(name_stem, datasets_cfg, extension):
    return name_stem + datasets_cfg["name_ext"] + "." + extension


def _get_filtered_runs(krr_cfg, ds, entity_name):
    if krr_cfg is None:
        return []

    project_name = krr_cfg["proj_name"] + ds
    runs = get_project_runs(entity_name, project_name)
    runs = filter_runs_union(runs, krr_cfg["criteria_list"])
    return runs


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

    n_rows, n_cols = _get_grid_shape(datasets_cfg)
    save_name = _get_save_name(name_stem, datasets_cfg, extension)

    for ds, config in datasets_cfg["datasets"].items():
        runs_full_krr = _get_filtered_runs(full_krr_cfg, ds, entity_name)
        runs_inducing_krr = _get_filtered_runs(inducing_krr_cfg, ds, entity_name)
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
