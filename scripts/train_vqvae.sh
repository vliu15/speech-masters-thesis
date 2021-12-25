#!/bin/bash

python train.py \
    --model vqvae \
    --dataset ljspeech \
    --log_dir ./logs/vqvae \
    --batch_size 8
