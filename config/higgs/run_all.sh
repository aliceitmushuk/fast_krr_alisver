#!/bin/bash

wandb_project_full=higgs_full_krr_v2
wandb_project_inducing=higgs_inducing_krr_v2

prefix="./config/higgs/"

inducing_krr_scripts=(
    "inducing_krr_falkon.sh"
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
bs=(2000 1000 500 200 100 50)

for precision in "${precisions[@]}"
do
    for script in "${inducing_krr_scripts[@]}"
    do
        bash "${prefix}${script}" "$precision" $wandb_project_inducing
    done
done

for precision in "${precisions[@]}"
do
    for script in "${full_krr_pcg_scripts[@]}"
    do
        bash "${prefix}${script}" "$precision" $wandb_project_full
    done
done

for b in "${bs[@]}"
do
    for script in "${full_krr_bcd_scripts[@]}"
    do
        bash "${prefix}${script}" "$b" $wandb_project_full
    done
done