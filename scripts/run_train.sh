#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1
# Define the command to run the training script with desired parameters
nohup python scripts/train.py \
    --modalities smiles text graph fp geom\
    --tasks tg er de td tm \
    --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
    > ./logs/train.log 2>&1 &
