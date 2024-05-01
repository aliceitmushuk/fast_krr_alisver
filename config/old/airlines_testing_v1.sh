#!/bin/bash

dataset=airlines
task=regression
sigma=4.0
lambd=1.0
opts=(bcd abcd)
b=900
ranks=(10 30 50)
max_iter=60000
log_freq=1000
seed=0
# devices=(0 1 2 3 4 5 6 7)
devices=(0 1 2 5 6 7)
wandb_project=airlines_testing_v1

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for opt in "${opts[@]}"
do
    for r in "${ranks[@]}"
    do
        device=${devices[counter]}
        python run_experiment.py --dataset $dataset --task $task --sigma $sigma --lambd $lambd --opt $opt --b $b --r $r --max_iter $max_iter \
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
