import os
from typing import Union

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import issparse, csr_matrix
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

from .data_configs import (
    DATA_DIR,
    DATA_CONFIGS,
    MOLECULES,
    SYNTHETIC_NTR,
    SYNTHETIC_NTST,
    SYNTHETIC_D,
)

LOADING_METHODS = {
    "libsvm": load_svmlight_file,
    "libsvm_multiple": load_svmlight_files,
    "h5py": h5py.File,
    "npz": np.load,
    "pkl": pd.read_pickle,
    "mat": loadmat,
}


def _standardize(data_tr: np.ndarray, data_tst: np.ndarray) -> tuple[np.ndarray]:
    reshaped = False

    # If data is one dimensional, reshape to 2D
    if len(data_tr.shape) == 1:
        reshaped = True
        data_tr = data_tr.reshape(-1, 1)
        data_tst = data_tst.reshape(-1, 1)

    scaler = StandardScaler()
    data_tr = scaler.fit_transform(data_tr)
    data_tst = scaler.transform(data_tst)

    if reshaped:
        data_tr = data_tr.flatten()
        data_tst = data_tst.flatten()

    return data_tr, data_tst


def _np_to_torch(
    X: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple[torch.Tensor]:
    X = torch.from_numpy(X)
    X = X.to(dtype=torch.get_default_dtype(), device=device)
    # X.requires_grad = True
    X.requires_grad = False
    y = torch.from_numpy(y)
    y = y.to(dtype=torch.get_default_dtype(), device=device)

    return X, y


def _np_to_torch_tr_tst(
    Xtr: np.ndarray,
    Xtst: np.ndarray,
    ytr: np.ndarray,
    ytst: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor]:
    Xtr, ytr = _np_to_torch(Xtr, ytr, device)
    Xtst, ytst = _np_to_torch(Xtst, ytst, device)

    return Xtr, Xtst, ytr, ytst


def _generate_synthetic_data(n: int, d: int) -> tuple[np.ndarray]:
    X = np.random.randn(n, d)
    w = np.random.randn(d)
    y = np.sign(X @ w)

    return X, y


def _map_labels(y: np.ndarray, label_map: dict) -> np.ndarray:
    return np.vectorize(label_map.get)(y)


def _one_hot_encode(X: pd.DataFrame, one_hot_cols: list[str]) -> pd.DataFrame:
    X_encoded = pd.get_dummies(X, columns=one_hot_cols)
    encoded_cols = X_encoded.columns.difference(X.columns)
    X_encoded[encoded_cols] = X_encoded[encoded_cols].astype(int)
    return X_encoded


def _process_molecule(R: np.ndarray) -> np.ndarray:
    n_atoms = R.shape[1]
    X = np.sum((R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
    X = X[:, np.triu_indices(n_atoms, 1)[0], np.triu_indices(n_atoms, 1)[1]] ** -1.0

    return X


def _convert_to_numpy(data: Union[csr_matrix, pd.DataFrame, np.ndarray]) -> np.ndarray:
    if issparse(data):
        return data.toarray()
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    else:
        return data


def load_data(dataset: str, seed: int, device: torch.device) -> tuple[torch.Tensor]:
    if dataset not in DATA_CONFIGS:
        raise ValueError(f"We do not currently support dataset {dataset}")

    data_config = DATA_CONFIGS[dataset]
    ftr = data_config.get("tr", None)
    ftst = data_config.get("tst", None)
    ftgt = data_config.get("tgt", None)
    loading_method = LOADING_METHODS[data_config["loading"]]

    X, Xtst, y, ytst = None, None, None, None

    # Load data, accounting for special cases
    if dataset == "synthetic":
        X, y = _generate_synthetic_data(SYNTHETIC_NTR + SYNTHETIC_NTST, SYNTHETIC_D)
    elif dataset == "qm9":
        data = loading_method(os.path.join(DATA_DIR, ftr))
        X, y = data["X"], data["Y"]
        y = np.squeeze(y)  # Remove singleton dimension due to .mat format
    elif dataset == "taxi_sub":
        with loading_method(os.path.join(DATA_DIR, ftr), "r") as f:
            X, y = f["X"][()], f["Y"][()]
        y = np.squeeze(y)
    # sgdml datasets
    elif dataset in MOLECULES:
        data = loading_method(os.path.join(DATA_DIR, ftr))
        X, y = _process_molecule(data["R"]), data["E"]
    else:
        # openml datasets
        if ftr is not None and ftgt is not None:
            X = loading_method(os.path.join(DATA_DIR, ftr))
            y = loading_method(os.path.join(DATA_DIR, ftgt))
        # libsvm datasets with train-test split
        elif ftr is not None and ftst is not None:
            X, y, Xtst, ytst = loading_method(
                [os.path.join(DATA_DIR, ftr), os.path.join(DATA_DIR, ftst)]
            )
        # libsvm datasets without train-test split
        elif ftr is not None:
            X, y = loading_method(os.path.join(DATA_DIR, ftr))

    # Label processing
    label_map = data_config.get("label_map", None)
    if label_map is not None:
        y = _map_labels(y, label_map)
        if ytst is not None:
            ytst = _map_labels(ytst, label_map)

    # One-hot encoding
    # only works for pandas DataFrames (basically just openml datasets)
    one_hot_cols = data_config.get("one_hot_cols", None)
    if one_hot_cols is not None:
        X = _one_hot_encode(X, one_hot_cols)
        if Xtst is not None:
            Xtst = _one_hot_encode(Xtst, one_hot_cols)

    # Turn sparse matrices and pandas DataFrames into numpy arrays
    X = _convert_to_numpy(X)
    if Xtst is not None:
        Xtst = _convert_to_numpy(Xtst)
    y = _convert_to_numpy(y)
    if ytst is not None:
        ytst = _convert_to_numpy(ytst)

    # Train-test split
    split = data_config.get("split", None)
    shuffle = data_config.get("shuffle", True)
    if split is not None:
        X, Xtst, y, ytst = train_test_split(
            X, y, train_size=split, random_state=seed, shuffle=shuffle
        )

    # Standardize
    X, Xtst = _standardize(X, Xtst)

    # Convert to torch tensors
    X, Xtst, y, ytst = _np_to_torch_tr_tst(X, Xtst, y, ytst, device)

    return X, Xtst, y, ytst


# # Modify to accomodate other datasets
# def load_data(dataset, seed, device):
#     if dataset == "synthetic":
#         Xtr, Xtst, ytr, ytst = _generate_synthetic_data(
#             SYNTHETIC_NTR, SYNTHETIC_NTST, SYNTHETIC_D
#         )
#     elif dataset == "homo":
#         data = loadmat(os.path.join(DATA_DIR, DATA_CONFIGS[dataset]))

#         X, y = data["X"], data["Y"]
#         y = np.squeeze(y)  # Remove singleton dimension due to .mat format

#         Xtr, Xtst, ytr, ytst = train_test_split(
#             X, y, train_size=100000, random_state=seed
#         )

#         Xtr, Xtst = standardize(Xtr, Xtst)
#     elif dataset == "susy":
#         data = load_svmlight_file(os.path.join(DATA_DIR, DATA_CONFIGS[dataset]))

#         X, y = data[0], data[1]
#         y[y == 0] = -1
#         X = X.toarray()

#         Xtr = X[:4500000]
#         Xtst = X[4500000:]
#         ytr = y[:4500000]
#         ytst = y[4500000:]

#         Xtr, Xtst = standardize(Xtr, Xtst)
#     elif dataset == "higgs":
#         data = load_svmlight_file(os.path.join(DATA_DIR, DATA_CONFIGS[dataset]))

#         X, y = data[0], data[1]
#         y[y == 0] = -1
#         X = X.toarray()

#         Xtr = X[:10500000]
#         Xtst = X[10500000:]
#         ytr = y[:10500000]
#         ytst = y[10500000:]

#         Xtr, Xtst = standardize(Xtr, Xtst)
#     elif dataset == "taxi_sub":
#         with h5py.File(os.path.join(DATA_DIR, DATA_CONFIGS[dataset]), "r") as f:
#             X = f["X"][()]
#             y = f["Y"][()]

#         y = np.squeeze(y)

#         Xtr, Xtst, ytr, ytst = train_test_split(
#             X, y, train_size=100_000_000, random_state=seed
#         )

#         Xtr, Xtst = standardize(Xtr, Xtst)
#     else:
#         raise ValueError("We do not currently support this dataset")

#     Xtr, Xtst, ytr, ytst = np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device)

#     return Xtr, Xtst, ytr, ytst
