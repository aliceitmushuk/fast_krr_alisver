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
    "a9a": {"tr": "a9a", "tst": "a9a.t"},
    "acsincome": {"tr": "acsincome_data.pkl", "tgt": "acsincome_target.pkl"},
    "airlines": {"tr": "airlines_data.pkl", "tgt": "airlines_target.pkl"},
    "aspirin": {"tr": "md17_aspirin.npz"},
    "benzene": {"tr": "md17_benzene2017.npz"},
    "cadata": {"tr": "cadata"},
    "click_prediction": {
        "tr": "click_prediction_data.pkl",
        "tgt": "click_prediction_target.pkl",
    },
    "cod_rna": {"tr": "cod-rna", "tst": "cod-rna.t"},
    "comet_mc": {"tr": "comet_mc_data.pkl", "tgt": "comet_mc_target.pkl"},
    "connect_4": {"tr": "connect-4"},
    "covtype_binary": "covtype.libsvm.binary.scale",
    "creditcard": {"tr": "creditcard_data.pkl", "tgt": "creditcard_target.pkl"},
    "diamonds": {"tr": "diamonds_data.pkl", "tgt": "diamonds_target.pkl"},
    "ethanol": {"tr": "md17_ethanol.npz"},
    "higgs": {"tr": "HIGGS"},
    "homo": {"tr": "homo.mat"},
    "hls4ml": {"tr": "hls4ml_data.pkl", "tgt": "hls4ml_target.pkl"},
    "ijcnn1": {"tr": "ijcnn1.tr", "tst": "ijcnn1.t"},
    "jannis": {"tr": "jannis_data.pkl", "tgt": "jannis_target.pkl"},
    "malonaldehyde": {"tr": "md17_malonaldehyde.npz"},
    "medical": {"tr": "medical_data.pkl", "tgt": "medical_target.pkl"},
    "miniboone": {"tr": "miniboone_data.pkl", "tgt": "miniboone_target.pkl"},
    "naphthalene": {"tr": "md17_naphthalene.npz"},
    "phishing": {"tr": "phishing"},
    "santander": {"tr": "santander_data.pkl", "tgt": "santander_target.pkl"},
    "salicylic": {"tr": "md17_salicylic.npz"},
    "sensit_vehicle": {"tr": "combined_scale", "tst": "combined_scale.t"},
    "sensorless": {"tr": "Sensorless.scale.tr", "tst": "Sensorless.scale.val"},
    "skin_nonskin": {"tr": "skin_nonskin"},
    "susy": {"tr": "SUSY"},
    "taxi_sub": {"tr": "taxi-data/subsampled_data.h5py"},
    "toluene": {"tr": "md17_toluene.npz"},
    "uracil": {"tr": "md17_uracil.npz"},
    "volkert": {"tr": "volkert_data.pkl", "tgt": "volkert_target.pkl"},
    "w8a": {"tr": "w8a", "tst": "w8a.t"},
    "yearpredictionmsd": {"tr": "YearPredictionMSD", "tst": "YearPredictionMSD.t"},
    "yolanda": {"tr": "yolanda_data.pkl", "tgt": "yolanda_target.pkl"},
}
DATA_KEYS = list(DATA_FILES.keys()) + ["synthetic"]
SYNTHETIC_NTR = 10000
SYNTHETIC_NTST = 1000
SYNTHETIC_D = 10


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


def _generate_synthetic_data(ntr, ntst, d):
    Xtr = np.random.randn(ntr, d)
    Xtst = np.random.randn(ntst, d)

    a = np.random.randn(d)
    ytr = np.sign(Xtr @ a)
    ytst = np.sign(Xtst @ a)
    return Xtr, Xtst, ytr, ytst


# Modify to accomodate other datasets
def load_data(dataset, seed, device):
    if dataset == "synthetic":
        Xtr, Xtst, ytr, ytst = _generate_synthetic_data(
            SYNTHETIC_NTR, SYNTHETIC_NTST, SYNTHETIC_D
        )
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
            X, y, train_size=100_000_000, random_state=seed
        )

        Xtr, Xtst = standardize(Xtr, Xtst)
        # ytr, ytst = standardize(ytr, ytst)
    else:
        raise ValueError("We do not currently support this dataset")

    Xtr, Xtst, ytr, ytst = np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device)

    return Xtr, Xtst, ytr, ytst
