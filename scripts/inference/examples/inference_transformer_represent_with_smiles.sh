#!/bin/bash

cd ../../..

source /fs1/software/modules/bashrc
module load loginnode
module load CUDA/11.8
module load cudnn/8.4.1-cuda11.x
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

SEED=2025
EPOCH=123

MODEL_CONFIG_PATH='./configs/config.json'
ENDPOINT_STATUS_FILE="./configs/tox_pc_mtl/qc_demo_status.json"

LATENT_SIZE=32
NUM_ENCODER_LAYERS=8
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1

HIDDEN_SIZE=512
NUM_LAYERS=3
DROPOUT_RATE=0.1

NUM_WORKERS=6
BETA=0.999
WEIGHT_DECAY=1e-2

DEVICE='cpu'

ENCODER_EVAL_FROM='./outs/checkpoints/pretrain/zinc22-10-gin/checkpoint_10.pt'
MODEL_EVAL_FROM="./outs/checkpoints/qc_demo_transformer_v3/100M_150/gin-L3-HS512-D0.4-ELR1e-4-LR1e-3-${SEED}-v3/checkpoint_${EPOCH}.pt"

TASK_TYPE='regression'

SMILES_LIST=$1

echo "=================="
representation=$(python inference_transformer.py \
  --seed $SEED \
  --task-type $TASK_TYPE \
  --model-config-path $MODEL_CONFIG_PATH \
  --endpoint-status-file $ENDPOINT_STATUS_FILE \
  --latent-size $LATENT_SIZE \
  --num-encoder-layers $NUM_ENCODER_LAYERS \
  --dropnode-rate $DROPNODE_RATE \
  --encoder-dropout $ENCODER_DROPOUT \
  --hidden-size $HIDDEN_SIZE \
  --num-layers $NUM_LAYERS \
  --dropout-rate $DROPOUT_RATE \
  --encoder-eval-from $ENCODER_EVAL_FROM \
  --model-eval-from $MODEL_EVAL_FROM \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --remove-hs \
  --smiles-list $SMILES_LIST \
  --representation | tail -n 1)
echo $representation