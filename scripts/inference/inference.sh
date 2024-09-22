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

LATENT_SIZE=128
ENCODER_HIDDEN_SIZE=512
NUM_ENCODER_LAYERS=2
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1
NUM_MESSAGE_PASSING_STEPS=2

HIDDEN_SIZE=512
NUM_LAYERS=3
DROPOUT_RATE=0.2

BATCH_SIZE=16
NUM_WORKERS=6

DEVICE='cpu'

SMILES_LIST=$1

TOXICITY_EVAL_FROM='./outs/checkpoints/downstream/toxicity/gat-3-0.2-1e-4-1e-2-frozen-beta/checkpoint_23.pt'
PHYSCHEM_EVAL_FROM='./outs/checkpoints/downstream/physchem-sample/gat-3-0.2-1e-4-1e-2-frozen-beta/checkpoint_87.pt'


python inference.py \
  --smiles-list $SMILES_LIST \
  --config-path $CONFIG_PATH \
  --status-path $STATUS_PATH \
  --latent-size $LATENT_SIZE \
  --encoder-hidden-size $ENCODER_HIDDEN_SIZE \
  --num-encoder-layers $NUM_ENCODER_LAYERS \
  --dropnode-rate $DROPNODE_RATE \
  --encoder-dropout $ENCODER_DROPOUT \
  --num-message-passing-steps $NUM_MESSAGE_PASSING_STEPS \
  --hidden-size $HIDDEN_SIZE \
  --num-layers $NUM_LAYERS \
  --dropout-rate $DROPOUT_RATE \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --toxicity-eval-from $TOXICITY_EVAL_FROM \
  --physchem-eval-from $PHYSCHEM_EVAL_FROM \
  --device $DEVICE \
  --remove-hs
