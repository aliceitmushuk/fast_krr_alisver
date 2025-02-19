# Fast KRR

<img src="logo.webp" width="200" height="200" alt="ASkotch Logo">

Companion code for "Have ASkotch: A Neat Solution for Large-scale Kernel Ridge Regression".

## Recreating the Python environment

Please create a Python virtual environment (our experiments use Python 3.10.12) and activate it. After activation, please run `pip install -r requirements.txt`
to download all required dependencies.

## Instructions for reproducing our experiments and figures

### Downloading the datasets for experiments

Running `download_data.py` will download all datasets we use in the paper, besides taxi.
The downloaded data will be placed in the `data` folder.

#### Obtaining the taxi dataset

Please clone the [nyc-taxi-data repo](https://github.com/pratikrathore8/nyc-taxi-data). Run `filter_runs.py` and `yellow_taxi_processing.sh` (NOTE: you may have to turn off the move to Google Drive step in this shell script) in the nyc-taxi-data repo.

This shell script will generate a `.h5py` file for each month from January 2009 to December 2015. Move these files to a new folder `data/taxi-data` and run `taxi_processing.py` in this (the fast_krr) repo.

### Running the experiments

We log the results of our experiments using Weights & Biases. To properly run the experiments, please create a Weights & Biases account and set up the API key.

#### Sections 6.1 and 6.4

To run the experiments in Sections 6.1 and 6.4, run `generate_configs_full_krr.py`, `generate_configs_eigenpro2.py`, `generate_configs_eigenpro3.py`, and `generate_configs_falkon.py`.
These scripts will generate configuration `.yaml` files in folders called `performance_full_krr`, `performance_full_krr_ep2`, `performance_inducing_krr_ep3`, and `performance_inducing_krr`.

To run each set of experiments, run `run_experiments.py` with the appropriate configuration folder as one of the arguments. For example:

```python
python run_experiments.py --base-dir performance_full_krr --devices 0 1 --grace-period-factor 0.4
```

The argument `--devices` specifies the GPU devices to use, and `--grace-period-factor` specifies the amount of extra time to run the method (since `run_experiments.py` does not account for time taken to perform inference on the test set during experiments).
Specifying multiple GPUs will run multiple experiments at the same time.
For example, `--devices 0 1` will run two experiments at the same time, one on GPU 0 and the other on GPU 1.

TODO: Add warning about the time taken to run the experiments.

#### Section 6.2

To run the experiments in Section 6.2, run `generate_configs_taxi.py`.
This script will generate configuration `.yaml` files in several folders that start with the word `taxi`.
Then run `run_experiments.py` with the appropriate arguments.

#### Section 6.3
To run the experiments in Section 6.3, run `generate_configs_lin_cvg.py`. This script will generate configuration `.yaml` files in the folder `lin_cvg_full_krr`.
Then run `run_experiments.py` with the appropriate arguments.

### Plotting

After running the experiments, plots can be generated using the Jupyter notebooks in `src/plotting`. You will have to change `ENTITY_NAME` in `plotting/constants.py` to your Weights & Biases entity name.
