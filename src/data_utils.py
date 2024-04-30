import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch

DATA_DIR = "./data/"
DATA_FILES = {
    "airlines": ["airlines_data.pkl", "airlines_target.pkl", "airport_to_atrcc.csv"],
    "homo": "homo.mat",
    "susy": "SUSY",
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
    elif dataset == "airlines":
        X = pd.read_pickle(os.path.join(DATA_DIR, DATA_FILES[dataset][0]))
        y = pd.read_pickle(os.path.join(DATA_DIR, DATA_FILES[dataset][1]))
        airport_to_atrcc = pd.read_csv(os.path.join(DATA_DIR, DATA_FILES[dataset][2]))

        # Transform Origin and Dest columns to ATRCC
        X["Origin"] = X["Origin"].map(airport_to_atrcc.set_index("AIRPORT")["ATRCC"])
        X["Dest"] = X["Dest"].map(airport_to_atrcc.set_index("AIRPORT")["ATRCC"])

        # Step 1: Identify unique carriers
        unique_carriers = pd.unique(X["UniqueCarrier"].values.ravel("K"))

        # Step 2: One-hot encode
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, categories=[unique_carriers])

        # Transform the carrier column
        carrier_encoded = encoder.fit_transform(X[["UniqueCarrier"]])

        # Convert the encoded features into DataFrame
        carrier_encoded_df = pd.DataFrame(
            carrier_encoded, columns=encoder.get_feature_names_out()
        )

        # Step 1: Identify unique ATRCCs
        unique_atrccs = pd.unique(X[["Origin", "Dest"]].values.ravel("K"))

        # Step 2: One-hot encode
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, categories=[unique_atrccs])

        # Transform both origin and destination using the same encoder
        origin_encoded = encoder.fit_transform(X[["Origin"]])
        destination_encoded = encoder.fit_transform(X[["Dest"]])

        # Convert the encoded features into DataFrame
        origin_encoded_df = pd.DataFrame(
            origin_encoded, columns=encoder.get_feature_names_out()
        )
        destination_encoded_df = pd.DataFrame(
            destination_encoded, columns=encoder.get_feature_names_out()
        )

        # Remove the original columns
        X = X.drop(columns=["Origin", "Dest", "UniqueCarrier"])

        # Concatenate the one-hot encoded features to the original DataFrame
        X = pd.concat(
            [X, carrier_encoded_df, origin_encoded_df, destination_encoded_df], axis=1
        )

        X = X.to_numpy()
        y = y.to_numpy()

        Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size=0.1, random_state=seed)

        Xtr_std_col, Xtst_std_col = standardize(Xtr[:, :6], Xtst[:, :6])
        Xtr[:, :6] = Xtr_std_col
        Xtst[:, :6] = Xtst_std_col

        ytr, ytst = standardize(ytr, ytst)
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
    else:
        raise ValueError("We do not currently support this dataset")

    Xtr, Xtst, ytr, ytst = np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device)

    return Xtr, Xtst, ytr, ytst
