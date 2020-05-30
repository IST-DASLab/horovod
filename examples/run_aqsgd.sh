#!/usr/bin/env bash
NAME="baseline"
CHECKPOINT_DIR="checkpoints/imagenet/$NAME/"

mkdir -p $CHECKPOINT_DIR

horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type uni --quantization-bits 32 --compression-bucket-size 128 \
python pytorch_imagenet_resnet50.py --model resnet18 --quantization-bits 3 --batch-size 128 --epochs 90 --train-dir /imagenet/train \
 --val-dir /imagenet/val --checkpoint-format "$CHECKPOINT_DIR/checkpoint-{epoch}.pth.tar" --enable-aqsgd 0 --bucket-size 128 2>&1 | tee logs/imagenet/$NAME

#horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type expL2 --quantization-bits 4 \
#python horovod_cifar.py --batch-size 256 --epochs 300 --dataset-dir /Datasets/cifar10 2>&1