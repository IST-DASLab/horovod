#!/usr/bin/env bash

create_line() {
    line="horovodrun -np $1 -H localhost:$1  python pytorch_synthetic_benchmark$2.py\
    --num-warmup-batches 10 --num-batches-per-iter 50 --num-iters 5 $3 2>&1 | tee scale_logs/sgd$2_$1"
}

for gpus in 2; do
    create_line $gpus
    eval $line
    create_line $gpus "_cross"
    eval $line
    create_line $gpus "_broken" "--bb-l2-ratio 0.0 --num-parallel-steps 5"
    eval $line
done
