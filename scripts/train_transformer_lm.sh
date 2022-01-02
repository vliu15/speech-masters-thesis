#!/bin/bash

python train.py \
    --model transformer_lm \
    --dataset vqlatent \
    --log_dir ./logs/transformer_lm_mmi \
    --ckpt_every_n_steps 2500 \
    --eval_every_n_epochs 2 \
    --batch_size 8
