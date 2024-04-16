#!/bin/bash

dataset=susy
task=classification
type=rbf
sigma=4.0
kernel_params="type $type sigma $sigma"
lambd=1e-3
opt=askotch
bs=(1 10 20 50 100 200 500 1000)
# beta=0
beta=1
precond_type=nystrom
r=100
precond_params="type $precond_type r $r"
max_iters=(20 200 400 1000 2000 4000 10000 20000)
log_freqs=(1 10 20 50 100 200 500 1000)
seed=0
devices=(0 1 2 3 4 5 6 7)
wandb_project=susy_testing_block_size

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for i in "${!bs[@]}"
do
    b=${bs[$i]}
    max_iter=${max_iters[$i]}
    log_freq=${log_freqs[$i]}

    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --task $task --kernel_params "$kernel_params" \
    --lambd $lambd --opt $opt --b $b --beta $beta --precond_params "$precond_params" \
    --max_iter $max_iter --log_freq $log_freq --seed $seed --device $device --wandb_project $wandb_project &
    counter=$((counter+1))
    # Ensure we don't exceed the number of devices
    if [ $counter -eq ${#devices[@]} ]; then
        counter=0
    fi
done

# Wait for all background processes to finish
wait
