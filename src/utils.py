import numpy as np
import random
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

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

# Modify to accomodate other datasets
def load_data(dataset, data_loc, device):
    if dataset == 'synthetic':
        Xtr = torch.randn(100000, 10, device=device)
        Xtst = torch.randn(10000, 10, device=device)
        Xtr.requires_grad = True
        Xtst.requires_grad = True

        a = torch.randn(10, device=device)
        ytr = torch.sign(Xtr @ a)
        ytst = torch.sign(Xtst @ a)
    elif dataset == 'susy':
        data = load_svmlight_file(data_loc)

        X, y = data[0], data[1]
        y[y == 0] = -1
        X = X.toarray()

        Xtr = X[:4500000]
        Xtst = X[4500000:]
        ytr = y[:4500000]
        ytst = y[4500000:]

        scaler = StandardScaler() 
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.transform(Xtst)

        Xtr = torch.from_numpy(Xtr).float().to(device)
        Xtst = torch.from_numpy(Xtst).float().to(device)
        Xtr.requires_grad = True
        Xtst.requires_grad = True

        ytr = torch.from_numpy(ytr).float().to(device)
        ytst = torch.from_numpy(ytst).float().to(device)
    else:
        raise ValueError('We do not support this dataset yet')

    return Xtr, Xtst, ytr, ytst