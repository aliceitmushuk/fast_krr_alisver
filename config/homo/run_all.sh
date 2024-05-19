#!/bin/bash

wandb_project_full=homo_full_krr_v2
wandb_project_inducing=homo_inducing_krr_v2

prefix="./config/homo/"

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
bs=(50 20 10 5 2 1)

full_krr_bcd_no_precond_scripts=(
    "full_krr_abcd.sh"
    "full_krr_bcd.sh"
)

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

for script in "${full_krr_bcd_no_precond_scripts[@]}"
do
    bash "${prefix}${script}" $wandb_project_full
done