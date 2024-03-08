#!/bin/bash

dataset=synthetic
task=classification
type=rbf
sigma=1.0
kernel_params="type $type sigma $sigma"
lambd=0.1
opts=(bcd abcd)
b=10
ranks=(10 30 50 100)
max_iter=30000
log_freq=100
seed=0
devices=(0 1 2 3 4 5 6 7)
wandb_project=synthetic_testing

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for opt in "${opts[@]}"
do
    for r in "${ranks[@]}"
    do
        device=${devices[counter]}
        python run_experiment.py --dataset $dataset --task $task --kernel_params "$kernel_params" --lambd $lambd --opt $opt --b $b --r $r --max_iter $max_iter \
                                    --log_freq $log_freq --seed $seed --device $device --wandb_project $wandb_project &
        counter=$((counter+1))
        # Ensure we don't exceed the number of devices
        if [ $counter -eq ${#devices[@]} ]; then
            counter=0
        fi
    done
done

# Wait for all background processes to finish
wait