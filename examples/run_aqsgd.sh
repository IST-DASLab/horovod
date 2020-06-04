#!/usr/bin/env bash
NAME="alq_L2_q3_b64"
DATASET=imagenet
CHECKPOINT_DIR="checkpoints/$DATASET/$NAME/"
mkdir -p $CHECKPOINT_DIR
mkdir -p logs/$DATASET
rm -rf ~/.horovod

horovodrun -np 2 -H localhost:2 --reduction-type Ring --compression-type expL2 --quantization-bits 3 --compression-bucket-size 64 \
python pytorch_imagenet_resnet50.py --model resnet18 --quantization-bits 3 --batch-size 32 --epochs 90 --train-dir /$DATASET/train \
 --val-dir /$DATASET/val --checkpoint-format "$CHECKPOINT_DIR/checkpoint-{epoch}.pth.tar" --enable-aqsgd 1 --bucket-size 64 2>&1 | tee logs/$DATASET/$NAME

#horovodrun -np 2 -H localhost:2 --reduction-type none --compression-type expL2 --quantization-bits 4 \
#python horovod_cifar.py --batch-size 256 --epochs 300 --dataset-dir /Datasets/cifar10 2>&1