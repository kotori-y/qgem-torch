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

SEED=$1
EPOCH=$2

MODEL_CONFIG_PATH='./configs/config.json'
# ENDPOINT_STATUS_FILE="./configs/em_statues/gmc_statues_${SEED}_all.json"
ENDPOINT_STATUS_FILE="./configs/tox_pc_mtl/toxicity_statues.json"

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
# MODEL_EVAL_FROM="./outs/checkpoints-all/gmc_mt_transformer_v3/100M_150/gin-L3-HS512-D0.1-ELR1e-4-LR1e-4-${SEED}-v3/"
MODEL_EVAL_FROM="./outs/checkpoints-all/toxicity_nova_v5_2_transformer_v3/100M_150/gin-L3-HS512-D0.1-ELR1e-4-LR1e-3-${SEED}-v3/checkpoint_${EPOCH}.pt"

# OUT_FILE="./outs/inference/extra_datav3_modelv3_555_all/extra_datav3_modelv3_SEED${SEED}_EPOCH${EPOCH}_all.csv"

TASK_TYPE='regression'

# CHECKPOINT_DIR="./outs/checkpoints_all/${NAME}_transformer/${OUT_DIR}/${MODEL_VER}-L${NUM_LAYERS}-HS${HIDDEN_SIZE}-D${DROPOUT_RATE}-ELR${ENCODER_LR}-LR${HEAD_LR}-${SEED}-v1"\

# ENDPOINTS="Decomposition Density Heat_of_Combustion Melting_Point"
ENDPOINTS="Cat_Intravenous_LD50 Chicken_Oral_LD50 Guineapig_Oral_LD50 Mouse_Intravenous_LD50 Mouse_Subcutaneous_LD50 Rabbit_Oral_LD50 Rat_Intravenous_LD50 Rat_Subcutaneous_LD50 Cat_Oral_LD50 Dog_Intravenous_LD50 Mouse_Intraperitoneal_LD50 Mouse_Oral_LD50 Rabbit_Intravenous_LD50 Rat_Intraperitoneal_LD50 Rat_Oral_LD50"

# SMILES_FILE="./data/demo/example.smi"
# OUT_FILE="./outs/inference/example.csv"
SMILES_FILE="./data/inference/toxgen_with_sascore_combined_drop_duplicated_lt_3_5_v2_.smi"
OUT_FILE="./outs/inference/toxgen_with_sascore_combined_drop_duplicated_lt_3_5_.csv"
# SMILES_FILE="./data/inference/demo.smi"

echo "run as single gpu mode"
echo "=================="
python inference_transformer.py \
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
  --endpoints $ENDPOINTS \
  --out-file $OUT_FILE \
  --smiles-file $SMILES_FILE
