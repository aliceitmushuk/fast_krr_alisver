#!/bin/bash

dataset=homo
model=full_krr
task=regression
kernel_type=l1_laplace
sigma=5120
kernel_params="type $kernel_type sigma $sigma"
lambd=1e-3
opt=askotch
bs=(50 20 10 5 2 1)
beta=0
max_time=3600
log_freq=50
precision=float32
seed=0
devices=(7 6 5 4 3 2 1 0)
wandb_project=$1

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for b in "${bs[@]}"
do
    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --model $model --task $task \
                            --kernel_params "$kernel_params" --lambd $lambd --opt $opt \
                            --b $b --beta $beta \
                            --max_time $max_time --log_freq $log_freq --precision $precision \
                            --seed $seed --device $device --wandb_project $wandb_project &
    counter=$((counter+1))
    # Ensure we don't exceed the number of devices
    if [ $counter -eq ${#devices[@]} ]; then
        counter=0
    fi
done

# Wait for all background processes to finish
wait