import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = '../data/'
DATA_FILES = {
    'airlines': ['airlines.data.pkl', 'airlines_target.pkl', 'airport_to_atrcc.csv'],
    'susy': 'SUSY'
}

"""
Helper function for setting seed for the random number generator in various packages.

INPUT: 
- seed: integer
"""
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def np_to_torch(X, y, device):
    X = torch.from_numpy(X).float().to(device)
    X.requires_grad = True
    y = torch.from_numpy(y).float().to(device)

    return X, y

def np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device):
    np_to_torch(Xtr, ytr, device)
    np_to_torch(Xtst, ytst, device)

    return Xtr, Xtst, ytr, ytst

# Modify to accomodate other datasets
def load_data(dataset, seed, device):
    if dataset == 'synthetic':
        Xtr = torch.randn(100000, 10, device=device)
        Xtst = torch.randn(10000, 10, device=device)
        Xtr.requires_grad = True
        Xtst.requires_grad = True

        a = torch.randn(10, device=device)
        ytr = torch.sign(Xtr @ a)
        ytst = torch.sign(Xtst @ a)
    elif dataset == 'airlines':
        X = pd.read_pickle(os.path.join(DATA_DIR, DATA_FILES[dataset][0]))
        y = pd.read_pickle(os.path.join(DATA_DIR, DATA_FILES[dataset][1]))
        airport_to_atrcc = pd.read_csv(os.path.join(DATA_DIR, DATA_FILES[dataset][2]))

        # Transform Origin and Dest columns to ATRCC
        X['Origin'] = X['Origin'].map(airport_to_atrcc.set_index('AIRPORT')['ATRCC'])
        X['Dest'] = X['Dest'].map(airport_to_atrcc.set_index('AIRPORT')['ATRCC'])

        # Step 1: Identify unique carriers
        unique_carriers = pd.unique(X['UniqueCarrier'].values.ravel('K'))

        # Step 2: One-hot encode
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, categories=[unique_carriers])

        # Transform the carrier column
        carrier_encoded = encoder.fit_transform(X[['UniqueCarrier']])

        # Convert the encoded features into DataFrame
        carrier_encoded_df = pd.DataFrame(
            carrier_encoded, columns=encoder.get_feature_names_out())

        # Step 1: Identify unique ATRCCs
        unique_atrccs = pd.unique(X[['Origin', 'Dest']].values.ravel('K'))

        # Step 2: One-hot encode
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, categories=[unique_atrccs])

        # Transform both origin and destination using the same encoder
        origin_encoded = encoder.fit_transform(X[['Origin']])
        destination_encoded = encoder.fit_transform(X[['Dest']])

        # Convert the encoded features into DataFrame
        origin_encoded_df = pd.DataFrame(
            origin_encoded, columns=encoder.get_feature_names_out())
        destination_encoded_df = pd.DataFrame(
            destination_encoded, columns=encoder.get_feature_names_out())

        # Remove the original columns
        X = X.drop(columns=['Origin', 'Dest', 'UniqueCarrier'])

        # Concatenate the one-hot encoded features to the original DataFrame
        X = pd.concat([X, carrier_encoded_df, origin_encoded_df,
                    destination_encoded_df], axis=1)

        X = X.to_numpy()
        y = y.to_numpy()

        Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size=0.1, random_state=seed)

        X_scaler = StandardScaler()
        Xtr[:, :6] = X_scaler.fit_transform(Xtr[:, :6]) # Only standardize non-one-hot encoded features
        Xtst[:, :6] = X_scaler.transform(Xtst[:, :6])

        y_scaler = StandardScaler()
        ytr = y_scaler.fit_transform(ytr.reshape(-1, 1)).flatten()
        ytst = y_scaler.transform(ytst.reshape(-1, 1)).flatten()

        Xtr, Xtst, ytr, ytst = np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device)
    elif dataset == 'susy':
        data = load_svmlight_file(os.path.join(DATA_DIR, DATA_FILES[dataset]))

        X, y = data[0], data[1]
        y[y == 0] = -1
        X = X.toarray()

        Xtr = X[:4500000]
        Xtst = X[4500000:]
        ytr = y[:4500000]
        ytst = y[4500000:]

        scaler = StandardScaler() 
        Xtr = scaler.fit_transform(Xtr)
        Xtst = scaler.transform(Xtst)

        Xtr, Xtst, ytr, ytst = np_to_torch_tr_tst(Xtr, Xtst, ytr, ytst, device)
    else:
        raise ValueError('We do not currently support this dataset')

    return Xtr, Xtst, ytr, ytst