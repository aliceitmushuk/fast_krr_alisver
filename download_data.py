import requests
import os
import subprocess
import bz2
import lzma
import shutil

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import torch
import qml
from scipy.io import savemat

from src.data_vision import process_all_datasets


def decompress_bz2(dataset, directory, file_path):
    print(f"Decompressing {dataset}...")
    dataset_trunc = dataset[:-4]
    new_file_path = os.path.join(directory, dataset_trunc)
    with bz2.BZ2File(file_path, "rb") as src, open(new_file_path, "wb") as dst:
        dst.write(src.read())
    print(f"Decompressed {dataset} successfully")


def decompress_xz(dataset, directory, file_path):
    print(f"Decompressing {dataset}...")
    dataset_trunc = dataset[:-3]
    new_file_path = os.path.join(directory, dataset_trunc)
    with lzma.open(file_path, "rb") as src, open(new_file_path, "wb") as dst:
        dst.write(src.read())
    print(f"Decompressed {dataset} successfully")


def download_openml(datasets, directory):
    for dataset in datasets:
        data, target = fetch_openml(data_id=dataset[1], return_X_y=True)
        pd.to_pickle(data, os.path.join(directory, f"{dataset[0]}_data.pkl"))
        pd.to_pickle(target, os.path.join(directory, f"{dataset[0]}_target.pkl"))
        print(f"Downloaded {dataset} successfully")


def download_libsvm(url_stem, datasets, directory):
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        url = f"{url_stem}/{dataset}"
        file_path = os.path.join(directory, dataset)

        # Download the dataset
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {dataset} successfully")

            # Decompress the dataset if the extension matches
            if dataset.endswith(".bz2"):
                decompress_bz2(dataset, directory, file_path)
            elif dataset.endswith(".xz"):
                decompress_xz(dataset, directory, file_path)
        else:
            print("Error: ", response.status_code)


def download_sgdml(url_stem, datasets, directory):
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        url = f"{url_stem}/{dataset}"
        file_path = os.path.join(directory, dataset)

        # Download the dataset
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {dataset} successfully")
        else:
            print("Error: ", response.status_code)


def download_qm9(url, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    cwd = os.getcwd()
    os.chdir(directory)

    # Download the file using wget, suppressing only stdout
    print("Downloading QM9 dataset...")
    subprocess.run(["wget", url], stdout=subprocess.DEVNULL)

    # Extract the contents of the file, suppressing only stdout
    print("Extracting QM9 dataset...")
    subprocess.run(["tar", "-xvf", "3195389"], stdout=subprocess.DEVNULL)

    # Remove the original downloaded file
    os.remove("3195389")

    # Change back to the previous directory
    os.chdir(cwd)


def process_qm9(directory, max_atoms=29, output_index=7):
    print("Processing QM9 dataset...")
    compounds = []
    energies = []
    for f in sorted(os.listdir(directory)):
        try:
            fname = os.path.join(directory, f)
            mol = qml.Compound(xyz=fname)
            mol.generate_coulomb_matrix(size=max_atoms, sorting="row-norm")
            with open(fname) as myfile:
                line = list(myfile.readlines())[1]
                energies.append(
                    float(line.split()[output_index]) * 27.2114
                )  # Hartrees to eV
            compounds.append(mol)
        except ValueError:
            pass
        finally:
            # Delete the file regardless of success or failure
            os.remove(fname)

    # After processing all files, delete the directory and its contents
    shutil.rmtree(directory)

    c = list(zip(compounds, energies))
    compounds, energies = zip(*c)

    X = np.array([mol.representation for mol in compounds])
    Y = np.array(energies).reshape((X.shape[0], 1))

    return X, Y


def main():
    # Create the data directory if it doesn't exist
    directory = os.path.abspath("./data")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # From LIBSVM
    url_stem = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
    datasets = [
        "covtype.libsvm.binary.scale.bz2",
        "HIGGS.xz",
        "SUSY.xz",
    ]
    download_libsvm(url_stem, datasets, directory)

    url_stem = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"
    datasets = ["YearPredictionMSD.bz2", "YearPredictionMSD.t.bz2"]
    download_libsvm(url_stem, datasets, directory)

    # From OpenML
    datasets = [
        ("acsincome", 43141),
        ("comet_mc", 23397),
        ("yolanda", 42705),
        ("click_prediction", 1218),
        ("miniboone", 41150),
    ]
    download_openml(datasets, directory)

    # From sGDML
    url_stem = "http://www.quantum-machine.org/gdml/data/npz"
    datasets = [
        "md17_benzene2017.npz",
        "md17_uracil.npz",
        "md17_naphthalene.npz",
        "md17_aspirin.npz",
        "md17_salicylic.npz",
        "md17_malonaldehyde.npz",
        "md17_ethanol.npz",
        "md17_toluene.npz",
    ]
    download_sgdml(url_stem, datasets, directory)

    # From QM9
    url = "https://figshare.com/ndownloader/files/3195389"
    directory_qm9 = os.path.join(directory, "qm9")
    download_qm9(url, directory_qm9)
    X, Y = process_qm9(directory_qm9)
    savemat(os.path.join(directory, "qm9.mat"), {"X": X, "Y": Y})

    # From torchvision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_all_datasets(directory, device)


if __name__ == "__main__":
    main()
