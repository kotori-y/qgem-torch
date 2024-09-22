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

DATASET_NAME=$2
CONFIG_PATH='./config/config.json'
DATASET_ROOT="./data/downstream/$DATASET_NAME"
DATASET_BASE_PATH="./data/downstream/$DATASET_NAME"

LATENT_SIZE=128
ENCODER_HIDDEN_SIZE=512
NUM_ENCODER_LAYERS=2
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1
NUM_MESSAGE_PASSING_STEPS=2

HIDDEN_SIZE=512
NUM_LAYERS=3
DROPOUT_RATE=0.2

BATCH_SIZE=128
NUM_WORKERS=6
BETA=0.999
WEIGHT_DECAY=1e-2

ENCODER_LR=1e-4
HEAD_LR=1e-2

EPOCHS=100
LOG_INTERVAL=5

DEVICE='cuda'
SEED=2024

MODEL_VER='gat'

CHECKPOINT_DIR="./outs/checkpoints/downstream/$DATASET_NAME/$MODEL_VER-$NUM_LAYERS-$DROPOUT_RATE-$ENCODER_LR-$HEAD_LR-frozen-beta"
ENCODER_EVAL_FROM='./outs/checkpoints/pretrain/demo-gat/checkpoint_5.pt'

TASK_TYPE='regr'

DISTRIBUTED=$1

ENDPOINTS="Cat_Intravenous_LD50 Cat_Oral_LD50 Chicken_Oral_LD50 Dog_Oral_LD50 Duck_Oral_LD50 Guineapig_Oral_LD50 Mouse_Intramuscular_LD50 Mouse_Intraperitoneal_LD50 Mouse_Intravenous_LD50 Mouse_Oral_LD50 Mouse_Subcutaneous_LD50 Rabbit_Intravenous_LD50 Rabbit_Oral_LD50 Rat_Inhalation_LC50 Rat_Intraperitoneal_LD50 Rat_Intravenous_LD50 Rat_Oral_LD50 Rat_Skin_LD50 Rat_Subcutaneous_LD50"

if [ $DISTRIBUTED -eq 0 ]; then
  echo "run as single gpu mode"
  echo "=================="
  python finetune_regr.py \
    --config-path $CONFIG_PATH \
    --dataset-root $DATASET_ROOT \
    --dataset-base-path $DATASET_BASE_PATH \
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
    --encoder-eval-from $ENCODER_EVAL_FROM \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --encoder_lr $ENCODER_LR \
    --head_lr $HEAD_LR \
    --beta $BETA \
    --weight-decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --log-interval $LOG_INTERVAL \
    --model-ver $MODEL_VER \
    --device $DEVICE \
    --lr-warmup \
    --use-adamw \
    --remove-hs \
    --endpoints $ENDPOINTS \
    --frozen-encoder
else
  echo "run as multi gpu mode"
  echo "=================="
  python -m torch.distributed.launch --nproc_per_node=2 pretrain.py \
    --distributed \
    --config-path $CONFIG_PATH \
    --dataset-root $DATASET_ROOT \
    --dataset-base-path $DATASET_BASE_PATH \
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
    --encoder-eval-from $ENCODER_EVAL_FROM \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --encoder_lr $ENCODER_LR \
    --head_lr $HEAD_LR \
    --beta $BETA \
    --weight-decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --log-interval $LOG_INTERVAL \
    --model-ver $MODEL_VER \
    --device $DEVICE \
    --lr-warmup \
    --use-adamw \
    --remove-hs \
    --endpoints $ENDPOINTS \
    --frozen-encoder
fi
