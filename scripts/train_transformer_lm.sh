#!/bin/bash

python train.py \
    --model transformer_lm \
    --dataset vqlatent \
    --log_dir ./logs/transformer_lm \
    --ckpt_every_n_steps 5000 \
    --eval_every_n_epochs 1 \
    --batch_size 4
