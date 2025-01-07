#!/bin/bash

# Array of Python scripts to run
scripts=(
    "askotch_ablation.py"
    "lin_cvg.py"
    "performance_comparison.py"
    "showcase.py"
)

# Function to handle cleanup when script is interrupted
cleanup() {
    echo "Interrupt received. Stopping all Python scripts..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Loop through each script and run it in the background
for script in "${scripts[@]}"
do
    echo "Starting $script..."
    python "$script" &  # Run the script in the background
done

# Wait for all background processes to finish
wait

echo "All scripts have finished running."
