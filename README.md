# Fast KRR

<img src="logo.webp" width="400" height="400" alt="SKOTCH Logo">

## Obtaining the Taxi dataset

Please clone [this GitHub repo](https://anonymous.4open.science/r/nyc-taxi-data). Then run `filter_runs.py` and `yellow_taxi_processing.sh` (NOTE: you may have to turn off the move to Google Drive step in this script) in this repo.

This will genrerate a `.h5py` file for each month from January 2009 to December 2015. Then, merge these files into a single `.h5py` file and place them in 
