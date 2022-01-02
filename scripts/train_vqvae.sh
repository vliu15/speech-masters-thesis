#!/bin/bash

# python train.py \
#     --model vqvae \
#     --dataset ljspeech \
#     --log_dir ./logs/vqvae \
#     --batch_size 4 \
#     --ckpt_every_n_steps 2500

python train.py \
    --model vqtts \
    --dataset ljspeech \
    --log_dir ./logs/vqtts \
    --batch_size 1 \
    --ckpt_every_n_steps 1000 \
    --run_sanity_val_epoch \
    --eval_every_n_epochs 2
