#!/usr/bin/env bash

horovodrun -np 2 -H localhost:2 --reduction-type Ring --compression-type expL2 --quantization-bits 4 \
python pytorch_imagenet_resnet50.py --quantization-bits 4 --batch-size 64 --epochs 90 --train-dir /Datasets/imagewoof/train \
 --val-dir /Datasets/imagewoof/val --checkpoint-format "checkpoints/imagewoof/alq/checkpoint-{epoch}.pth.tar" 2>&1 | tee conv_logs/imagewoof/amq_nb_q4

#horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type expL2 --quantization-bits 4 \
#python horovod_cifar.py --batch-size 256 --epochs 300 --dataset-dir /Datasets/cifar10 2>&1