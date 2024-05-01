#!/bin/bash

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

for script in "${inducing_krr_scripts[@]}"
do
    bash "${prefix}${script}"
done

for precision in "${precisions[@]}"
do
    for script in "${full_krr_pcg_scripts[@]}"
    do
        bash "${prefix}${script}" "$precision"
    done
done

for b in "${bs[@]}"
do
    for script in "${full_krr_bcd_scripts[@]}"
    do
        bash "${prefix}${script}" "$b"
    done
done