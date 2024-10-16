#!/bin/bash

wandb_project_full=taxi_full_krr
wandb_project_inducing=taxi_inducing_krr
max_time=86400

prefix="./config/taxi/"

inducing_krr_falkon_scripts=(
    "inducing_krr_falkon.sh"
)

inducing_krr_sketchy_scripts=(
    "inducing_krr_saga.sh"
    "inducing_krr_sksaga.sh"
)

full_krr_pcg_scripts=(
    "full_krr_chol_pcg.sh"
    "full_krr_nystrom_pcg.sh"
)
precisions=(float32 float64)

full_krr_bcd_scripts=(
    "full_krr_askotch.sh"
    "full_krr_skotch.sh"
)

for precision in "${precisions[@]}"
do
    for script in "${inducing_krr_falkon_scripts[@]}"
    do
        bash "${prefix}${script}" "$precision" $wandb_project_inducing $max_time
    done
done

for script in "${full_krr_bcd_scripts[@]}"
do
    bash "${prefix}${script}" $wandb_project_full $max_time
done

for precision in "${precisions[@]}"
do
    for script in "${full_krr_pcg_scripts[@]}"
    do
        bash "${prefix}${script}" "$precision" $wandb_project_full $max_time
    done
done

for script in "${inducing_krr_sketchy_scripts[@]}"
do
    bash "${prefix}${script}" $wandb_project_inducing $max_time
done