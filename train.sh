#!/usr/bin/env bash

echo $BASHPID

MAX_PARALLEL=1
N_CONFIGS=3

run_job () {
    python -u src/train.py "$1"
}

for ((i=0; i<N_CONFIGS; i++)); do
    run_job "$i" &
    if (( $(jobs -r | wc -l) >= MAX_PARALLEL )); then
        wait -n
    fi
done

wait