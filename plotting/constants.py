# high-level plotting parameters
EXTENSION = "pdf"
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
        "yolanda": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "yearpredictionmsd": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "acsincome": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 3},
    "name_ext": "tabular_regression",
}
ALL_PERFORMANCE_DATASETS = [
    VISION,
    PARTICLE_PHYSICS,
    TABULAR_CLASSIFICATION,
    QM9,
    MOLECULES_BIG,
    MOLECULES_SMALL,
    TABULAR_REGRESSION,
]
