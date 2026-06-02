#!/bin/bash

# Fixed args
#COMMON_ARGS="--dataset waterbirds --model_arch ViT --output_dir output --img_size 224 --eval_every 100 --num_steps 1000 --max_grad_norm 1.0 --eval_batch_size 64"

CUDA_VISIBLE_DEVICES=0,1,3,4,5,7 accelerate launch train_accelerate.py \
  --name camelyon17-set2_exp --model_arch ViT --dataset camelyon17-set2 --num_steps 1000 --batch_split 1 --img_size 224 --eval_every 100

