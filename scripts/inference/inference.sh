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

CONFIG_PATH='./configs/config.json'
STATUS_PATH='./configs/endpoint_statuses.json'

LATENT_SIZE=32
NUM_ENCODER_LAYERS=8
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1

HIDDEN_SIZE_TOX=128
HIDDEN_SIZE_PC=512
NUM_LAYERS=3
DROPOUT_RATE=0.2

BATCH_SIZE=128
NUM_WORKERS=6

DEVICE='cuda'

SMILES_LIST=$1

TOXICITY_EVAL_FROM='./outs/checkpoints/downstream/toxicity-wash-3/gin-L3-D0.2-ELR1e-4-LR1e-3/checkpoint_99.pt'
PHYSCHEM_EVAL_FROM='./outs/checkpoints/downstream/physchem-wash-2/gin-L3-D0.2-ELR1e-4-LR1e-3/checkpoint_22.pt'


python inference.py \
  --smiles-list $SMILES_LIST \
  --config-path $CONFIG_PATH \
  --status-path $STATUS_PATH \
  --latent-size $LATENT_SIZE \
  --num-encoder-layers $NUM_ENCODER_LAYERS \
  --dropnode-rate $DROPNODE_RATE \
  --encoder-dropout $ENCODER_DROPOUT \
  --hidden-size-pc $HIDDEN_SIZE_PC \
  --hidden-size-tox $HIDDEN_SIZE_TOX \
  --num-layers $NUM_LAYERS \
  --dropout-rate $DROPOUT_RATE \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --toxicity-eval-from $TOXICITY_EVAL_FROM \
  --physchem-eval-from $PHYSCHEM_EVAL_FROM \
  --device $DEVICE \
  --remove-hs
