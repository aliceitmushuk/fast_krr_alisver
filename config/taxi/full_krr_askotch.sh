#!/bin/bash

dataset=taxi_sub
model=full_krr
task=regression
kernel_type=rbf
sigma=1.0
kernel_params="type $kernel_type sigma $sigma"
lambd=19.8
opt=askotch
b=2000
beta=0
precond_type=nystrom
ranks=(50 100 200 500)
max_time=18000
log_freq=20
precision=float32
seed=0
devices=(7 0 2 3)
wandb_project=taxi_full_krr_testing

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for r in "${ranks[@]}"
do
    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --model $model --task $task \
                            --kernel_params "$kernel_params" --lambd $lambd --opt $opt \
                            --b $b --beta $beta --no_store_precond --precond_params "type $precond_type r $r" \
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