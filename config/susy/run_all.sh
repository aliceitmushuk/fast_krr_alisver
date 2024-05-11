#!/bin/bash

wandb_project=susy_full_krr_v2

prefix="./config/susy/"

full_krr_pcg_scripts=(
    "full_krr_chol_pcg.sh"
    "full_krr_nystrom_pcg.sh"
)
precisions=(float32 float64)

full_krr_bcd_scripts=(
    "full_krr_askotch.sh"
    "full_krr_skotch.sh"
)
bs=(1000 500 200 100 50 20 10 5 2 1)

for precision in "${precisions[@]}"
do
    for script in "${full_krr_pcg_scripts[@]}"
    do
        bash "${prefix}${script}" "$precision" $wandb_project
    done
done

for b in "${bs[@]}"
do
    for script in "${full_krr_bcd_scripts[@]}"
    do
        bash "${prefix}${script}" "$b" $wandb_project
    done
done