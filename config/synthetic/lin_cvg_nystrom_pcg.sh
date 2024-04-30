#!/bin/bash

dataset=synthetic
model=full_krr
task=regression
kernel_type=rbf
sigma=1.0
kernel_params="type $kernel_type sigma $sigma"
lambd=0.1
opt=pcg
precond_type=nystrom
ranks=(10 30 50 100)
max_iter=3000
log_freq=10
precision=float64
seed=0
devices=(0 2 3 4)
wandb_project=linear_convergence_full_krr

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for r in "${ranks[@]}"
do
    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --model $model --task $task \
                            --kernel_params "$kernel_params" --lambd $lambd --opt $opt \
                            --precond_params "type $precond_type r $r rho $lambd" \
                            --max_iter $max_iter --log_freq $log_freq --precision $precision \
                            --seed $seed --device $device --wandb_project $wandb_project &
    counter=$((counter+1))
    # Ensure we don't exceed the number of devices
    if [ $counter -eq ${#devices[@]} ]; then
        counter=0
    fi
done

# Wait for all background processes to finish
wait