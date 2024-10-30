#!/bin/bash

dataset=taxi_sub
model=inducing_krr
task=regression
kernel_type=rbf
sigma=1.0
kernel_params="type $kernel_type sigma $sigma"
m=1000000
lambd=20
opt=sketchysaga
bg=65536
precond_type=nystrom
r=300
rhos=(1000000 3000000 10000000 30000000 100000000)
max_time=$2
log_freq=20
precision=float64
seed=0
devices=(0 1 2 3 4)
wandb_project=$1

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

for rho in "${rhos[@]}"
do
    device=${devices[counter]}
    python run_experiment.py --dataset $dataset --model $model --task $task \
                            --kernel_params "$kernel_params" --m $m --lambd $lambd --opt $opt \
                            --bg $bg --precond_params "type $precond_type r $r rho $rho use_cpu" \
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
