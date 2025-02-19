#!/bin/bash
export PYTHONPATH=$(pwd)
base_dir="config_gen"

# Array of Python scripts to run
scripts=(
    "eigenpro2.py"
    "eigenpro3.py"
    "falkon.py"
    "full_krr.py"
    "lin_cvg.py"
    "taxi.py"
)


# Loop through each script
for script in "${scripts[@]}"
do
    script_path="$base_dir/$script"
    echo "Starting $script_path..."
    python "$script_path"
done
