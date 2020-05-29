#!/usr/bin/env bash
NAME="alq_nb_q3_bs8192"
CHECKPOINT_DIR="checkpoints/imagewoof/$NAME/"

mkdir -p $CHECKPOINT_DIR

horovodrun -np 2 -H localhost:2 --reduction-type ScatterAllgather --compression-type expL2 --quantization-bits 4 --compression-bucket-size 8192 \
python pytorch_imagenet_resnet50.py --model resnet18 --quantization-bits 4 --batch-size 32 --epochs 90 --train-dir /imagewoof/train \
 --val-dir /imagewoof/val --checkpoint-format "$CHECKPOINT_DIR/checkpoint-{epoch}.pth.tar" --enable-aqsgd 1 2>&1 | tee logs/imagewoof/$NAME

#horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type expL2 --quantization-bits 4 \
#python horovod_cifar.py --batch-size 256 --epochs 300 --dataset-dir /Datasets/cifar10 2>&1