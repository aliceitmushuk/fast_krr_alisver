#!/bin/bash

dataset=synthetic
model=inducing_krr
task=regression
kernel_type=rbf
sigma=1.0
kernel_params="type $kernel_type sigma $sigma"
m=1000
lambd=0.1
opt=sketchysaga
bg=1024
precond_type=nystrom
ranks=(50 100 200 500)
rho=1
max_iter=100000
log_freq=10
precision=float64
seed=0
devices=(0 2 3 4)
wandb_project=$1

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for r in "${ranks[@]}"
do
    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --model $model --task $task \
                            --kernel_params "$kernel_params" --m $m --lambd $lambd --opt $opt \
                            --bg $bg --precond_params "type $precond_type r $r rho $rho" \
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
