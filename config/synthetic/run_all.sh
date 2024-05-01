#!/bin/bash

prefix="./config/synthetic/"

scripts=(
    "lin_cvg_skotch.sh"
    "lin_cvg_askotch.sh"
    "lin_cvg_nystrom_pcg.sh"
    "lin_cvg_sksaga.sh"
    "lin_cvg_skkat.sh"
    "lin_cvg_falkon.sh"
)

for script in "${scripts[@]}"; do
    bash "${prefix}${script}"
done