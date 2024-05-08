import os

import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

DATA_DIR = "./data/"
DATA_FILES = {
    "homo": "homo.mat",
    "susy": "SUSY",
    "higgs": "HIGGS",
    "taxi_sub": "taxi-data/subsampled_data.h5py"
}


def standardize(data_tr, data_tst):
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


def np_to_torch(X, y, device):
    X = torch.from_numpy(X)
    X = X.to(dtype=torch.get_default_dtype(), device=device)
    # X.requires_grad = True
    X.requires_grad = False
    y = torch.from_numpy(y)
    y = y.to(dtype=torch.get_default_dtype(), device=device)

    return X, y


def np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device):
    Xtr, ytr = np_to_torch(Xtr, ytr, device)
    Xtst, ytst = np_to_torch(Xtst, ytst, device)

    return Xtr, Xtst, ytr, ytst


# Modify to accomodate other datasets
def load_data(dataset, seed, device):
    if dataset == "synthetic":
        Xtr = np.random.randn(10000, 10)
        Xtst = np.random.randn(1000, 10)

        a = np.random.randn(10)
        ytr = np.sign(Xtr @ a)
        ytst = np.sign(Xtst @ a)
    elif dataset == "homo":
        data = loadmat(os.path.join(DATA_DIR, DATA_FILES[dataset]))

        X, y = data["X"], data["Y"]
        y = np.squeeze(y)  # Remove singleton dimension due to .mat format

        Xtr, Xtst, ytr, ytst = train_test_split(
            X, y, train_size=100000, random_state=seed
        )

        Xtr, Xtst = standardize(Xtr, Xtst)
        # ytr, ytst = standardize(ytr, ytst)
    elif dataset == "susy":
        data = load_svmlight_file(os.path.join(DATA_DIR, DATA_FILES[dataset]))

        X, y = data[0], data[1]
        y[y == 0] = -1
        X = X.toarray()

        Xtr = X[:4500000]
        Xtst = X[4500000:]
        ytr = y[:4500000]
        ytst = y[4500000:]

        Xtr, Xtst = standardize(Xtr, Xtst)
    elif dataset == "higgs":
        data = load_svmlight_file(os.path.join(DATA_DIR, DATA_FILES[dataset]))

        X, y = data[0], data[1]
        y[y == 0] = -1
        X = X.toarray()

        Xtr = X[:10500000]
        Xtst = X[10500000:]
        ytr = y[:10500000]
        ytst = y[10500000:]

        Xtr, Xtst = standardize(Xtr, Xtst)
    elif dataset == "taxi_sub":
        with h5py.File(os.path.join(DATA_DIR, DATA_FILES[dataset]), "r") as f:
            X = f["X"][()]
            y = f["Y"][()]

        y = np.squeeze(y)

        Xtr, Xtst, ytr, ytst = train_test_split(
            X, y, train_size=0.999, random_state=seed
        )

        Xtr, Xtst = standardize(Xtr, Xtst)
        # ytr, ytst = standardize(ytr, ytst)
    else:
        raise ValueError("We do not currently support this dataset")

    Xtr, Xtst, ytr, ytst = np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device)

    return Xtr, Xtst, ytr, ytst
