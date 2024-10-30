#!/bin/bash

wandb_project_full=linear_convergence_full_krr_v2
wandb_project_inducing=linear_convergence_inducing_krr_v2

prefix="./config/synthetic/"

inducing_krr_scripts=(
    "lin_cvg_sksaga.sh"
    "lin_cvg_skkat.sh"
    "lin_cvg_falkon.sh"
)

full_krr_scripts=(
    "lin_cvg_skotch.sh"
    "lin_cvg_askotch.sh"
    "lin_cvg_nystrom_pcg.sh"
)

for script in "${full_krr_scripts[@]}"; do
    bash "${prefix}${script}" $wandb_project_full
done

for script in "${inducing_krr_scripts[@]}"; do
    bash "${prefix}${script}" $wandb_project_inducing
done
