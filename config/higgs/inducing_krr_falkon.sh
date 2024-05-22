#!/bin/bash

dataset=higgs
model=inducing_krr
task=classification
kernel_type=rbf
sigma=3.8
kernel_params="type $kernel_type sigma $sigma"
ms=(500 1000 2000 5000 10000 20000 50000 100000)
lambd=0.315
opt=pcg
precond_type=falkon
max_time=7200
log_freq=50
precision=$1
seed=0
devices=(7 6 5 4 3 2 1 0)
wandb_project=$2

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for m in "${ms[@]}"
do
    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --model $model --task $task \
                            --kernel_params "$kernel_params" --m $m --lambd $lambd --opt $opt \
                            --precond_params "type $precond_type" \
                            --max_time $max_time --log_freq $log_freq --log_test_only --precision $precision \
                            --seed $seed --device $device --wandb_project $wandb_project &
    counter=$((counter+1))
    # Ensure we don't exceed the number of devices
    if [ $counter -eq ${#devices[@]} ]; then
        counter=0
    fi
done

# Wait for all background processes to finish
wait