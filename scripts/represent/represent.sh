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
STATUS_PATH='./configs/quandb_statuses.json'

LATENT_SIZE=32
NUM_ENCODER_LAYERS=8
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1

HIDDEN_SIZE=128
NUM_LAYERS=3
DROPOUT_RATE=0.4

BATCH_SIZE=128
NUM_WORKERS=6

DEVICE='cpu'

SMILES_LIST=$1

ENDPOINTS="zpe e0K eOpt U H S G cv uCorre hCorre gCorre eHomoOpt eLumoOpt eGapOpt eseOpt muOpt thetaOpt alphaOpt volumeMC e eHOMO eLUMO eGap ese mu theta espcMax hirshfeldMax cm5Max npaMax espcMin hirshfeldMin cm5Min npaMin volumeIMT density sa saPos saNeg saNonpolar saPolar espAvg espAvgPos espAvgNeg espVar espVarPos espVarNeg espMax espMin nu productOfSigmaSquareAndNu pi mpi"

EVAL_FROM="./outs/checkpoints/molnet/QGEM/100M/quandb/gin-L3-D0.4-ELR1e-3-LR1e-3-BS128-HS128/checkpoint_77.pt"

python represent.py \
  --smiles-list $SMILES_LIST \
  --config-path $CONFIG_PATH \
  --status-path $STATUS_PATH \
  --latent-size $LATENT_SIZE \
  --num-encoder-layers $NUM_ENCODER_LAYERS \
  --dropnode-rate $DROPNODE_RATE \
  --encoder-dropout $ENCODER_DROPOUT \
  --hidden-siz $HIDDEN_SIZE \
  --num-layers $NUM_LAYERS \
  --dropout-rate $DROPOUT_RATE \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --endpoints $ENDPOINTS \
  --eval-from $EVAL_FROM \
  --remove-hs
