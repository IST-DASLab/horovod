#!/usr/bin/env bash

horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type expL2 --quantization-bits 32 \
python pytorch_imagenet_resnet50.py --quantization-bits 4 --batch-size 64 --epochs 90 --train-dir /imagenet/train \
 --val-dir /imagenet/val --checkpoint-format "checkpoints/imagenet/baseline_rn18/checkpoint-{epoch}.pth.tar" 2>&1 | tee imagenet/rn18_baseline

#horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type expL2 --quantization-bits 4 \
#python horovod_cifar.py --batch-size 256 --epochs 300 --dataset-dir /Datasets/cifar10 2>&1