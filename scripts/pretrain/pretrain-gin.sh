#!/bin/bash

cd ../..

source /fs1/software/modules/bashrc
module load loginnode
module load CUDA/11.8
module load cudnn/8.4.1-cuda11.x
export PYTHONUNBUFFERED=1

# root_path="../../../.."
# export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

CONFIG_PATH='./config.json'
DATASET_NAME=$1
DATASET_ROOT='./data/'$DATASET_NAME
DATASET_BASE_PATH='./data/'$DATASET_NAME

MASK_RATIO=0.15

LATENT_SIZE=256
ENCODER_HIDDEN_SIZE=1024
NUM_ENCODER_LAYERS=2
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1
NUM_MESSAGE_PASSING_STEPS=2

HIDDEN_SIZE=1024
NUM_LAYERS=2
DROPOUT_RATE=0.1

BATCH_SIZE=128
NUM_WORKERS=72
LR=1e-4
BETA=0.999
WEIGHT_DECAY=1e-2

EPOCHS=100
LOG_INTERVAL=5

DEVICE='cuda'
SEED=2024

MODEL_VER='gin'

CHECKPOINT_DIR='./outs/checkpoints/'$DATASET_NAME'-'$MODEL_VER

GPU_NODE='gpu3'
GPU_NUM=2

python pretrain.py \
  --config-path $CONFIG_PATH \
  --dataset-root $DATASET_ROOT \
  --dataset-base-path $DATASET_BASE_PATH \
  --dataset-name $DATASET_NAME \
  --mask-ratio $MASK_RATIO \
  --latent-size $LATENT_SIZE \
  --encoder-hidden-size $ENCODER_HIDDEN_SIZE \
  --num-encoder-layers $NUM_ENCODER_LAYERS \
  --dropnode-rate $DROPNODE_RATE \
  --encoder-dropout $ENCODER_DROPOUT \
  --num-message-passing-steps $NUM_MESSAGE_PASSING_STEPS \
  --hidden-size $HIDDEN_SIZE \
  --num-layers $NUM_LAYERS \
  --dropout-rate $DROPOUT_RATE \
  --checkpoint-dir $CHECKPOINT_DIR \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --beta $BETA \
  --weight-decay $WEIGHT_DECAY \
  --epochs $EPOCHS \
  --log-interval $LOG_INTERVAL \
  --model-ver $MODEL_VER \
  --device $DEVICE \
  --lr-warmup \
  --use-adamw \
  --remove-hs
