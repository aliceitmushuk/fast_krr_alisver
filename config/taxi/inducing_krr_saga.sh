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
max_time=$2
log_freq=20
precision=float64
seed=0
device=0
wandb_project=$1

# Initialize the counter
counter=0

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

python run_experiment.py --dataset $dataset --model $model --task $task \
                        --kernel_params "$kernel_params" --m $m --lambd $lambd --opt $opt \
                        --bg $bg \
                        --max_time $max_time --log_freq $log_freq --log_test_only --precision $precision \
                        --seed $seed --device $device --wandb_project $wandb_project &

# Wait for all background processes to finish
wait
