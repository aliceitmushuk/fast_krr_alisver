from utils import get_project_runs, filter_runs_multi


def get_grid_shape(datasets_cfg):
    n_rows = datasets_cfg["grid"]["n_rows"]
    n_cols = datasets_cfg["grid"]["n_cols"]
    return n_rows, n_cols


def get_save_name(name_stem, datasets_cfg, extension):
    return name_stem + datasets_cfg["name_ext"] + "." + extension


def get_filtered_krr_runs(krr_cfg, ds, entity_name):
    project_name = krr_cfg["proj_name"] + ds
    runs = get_project_runs(entity_name, project_name)
    runs = filter_runs_multi(runs, krr_cfg["criteria_list"])
    return runs
