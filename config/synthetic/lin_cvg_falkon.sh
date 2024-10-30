#!/bin/bash

dataset=synthetic
model=inducing_krr
task=regression
kernel_type=rbf
sigma=1.0
kernel_params="type $kernel_type sigma $sigma"
m=1000
lambd=0.1
opt=pcg
precond_type=falkon
max_iter=1000
log_freq=10
precision=float64
seed=0
device=0
wandb_project=$1

# Trap SIGINT (Ctrl-C) and SIGTERM to kill child processes
trap "kill 0" SIGINT SIGTERM

python run_experiment.py --dataset $dataset --model $model --task $task \
                        --kernel_params "$kernel_params" --m $m --lambd $lambd --opt $opt \
                        --precond_params "type $precond_type" \
                        --max_iter $max_iter --log_freq $log_freq --precision $precision \
                        --seed $seed --device $device --wandb_project $wandb_project &

# Wait for all background processes to finish
wait
