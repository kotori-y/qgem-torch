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

SEED=$8

MODEL_CONFIG_PATH='./configs/config.json'
# ENDPOINT_STATUS_FILE='./configs/tox_pc_mtl/toxicity_cls.json'
# ENDPOINT_STATUS_FILE='./configs/tox_pc_mtl/physchem.json'
ENDPOINT_STATUS_FILE='./configs/tox_pc_mtl/qc.json'

# NAME="toxicity_nova_v2_cls"
# NAME="physchem_nova_v2"
NAME="qc_demo"
DATASET_BASE_PATH="./data/downstream/${NAME}"

LATENT_SIZE=32
NUM_ENCODER_LAYERS=8
DROPNODE_RATE=0.1
ENCODER_DROPOUT=0.1

HIDDEN_SIZE=512
NUM_LAYERS=3
DROPOUT_RATE=$6

BATCH_SIZE=$7
NUM_WORKERS=6
BETA=0.999
WEIGHT_DECAY=1e-2

ENCODER_LR=$4
HEAD_LR=$5

EPOCHS=150
LOG_INTERVAL=5

DEVICE='cuda'

MODEL_VER='gin'

ENCODER_EVAL_FROM=$2

TASK_TYPE='regression'
N_LABELS=5

DISTRIBUTED=$1
OUT_DIR=$3

CHECKPOINT_DIR="./outs/checkpoints/${NAME}_transformer_v3/${OUT_DIR}_${EPOCHS}/${MODEL_VER}-L${NUM_LAYERS}-HS${HIDDEN_SIZE}-D${DROPOUT_RATE}-ELR${ENCODER_LR}-LR${HEAD_LR}-${SEED}-v3"
# ENDPOINTS="Cat_Intravenous_LD50 Chicken_Oral_LD50 Guineapig_Oral_LD50 Mouse_Intravenous_LD50 Mouse_Subcutaneous_LD50 Rabbit_Oral_LD50 Rat_Intravenous_LD50 Rat_Subcutaneous_LD50 Cat_Oral_LD50 Dog_Intravenous_LD50 Mouse_Intraperitoneal_LD50 Mouse_Oral_LD50 Rabbit_Intravenous_LD50 Rat_Intraperitoneal_LD50 Rat_Oral_LD50"
# ENDPOINTS="Boiling_Point Density Flash_Point LogP LogS Melting_Point Refractive_Index Surface_Tension Vapor_Pressure"
# ENDPOINTS="Rat_Oral_LD50 Rat_Subcutaneous_LD50"
ENDPOINTS="alphaOpt density eGapOpt eLumoOpt espAvg espcMin espVarNeg hCorre mpi npaMin sa saPos uCorre zpe cm5Max e0K eHOMO eOpt espAvgNeg espMax espVarPos H mu nu saNeg S U cm5Min e eHomoOpt ese espAvgPos espMin gCorre hirshfeldMax muOpt pi saNonpolar theta volumeIMT cv eGap eLUMO eseOpt espcMax espVar G hirshfeldMin npaMax productOfSigmaSquareAndNu saPolar thetaOpt volumeMC"

PREPROCESS_ENDPOINTS="kotori"
EXCLUDE_SMILES="kotori"

if [ $DISTRIBUTED -eq 0 ]; then
  echo "run as single gpu mode"
  echo "=================="
  python finetune_transformer.py \
    --seed $SEED \
    --model-config-path $MODEL_CONFIG_PATH \
    --endpoint-status-file $ENDPOINT_STATUS_FILE \
    --dataset-base-path $DATASET_BASE_PATH \
    --latent-size $LATENT_SIZE \
    --num-encoder-layers $NUM_ENCODER_LAYERS \
    --dropnode-rate $DROPNODE_RATE \
    --encoder-dropout $ENCODER_DROPOUT \
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
    --enable-tb \
    --endpoints $ENDPOINTS \
    --preprocess-endpoints $PREPROCESS_ENDPOINTS \
    --exclude-smiles $EXCLUDE_SMILES \
    --task-type $TASK_TYPE \
    --n-labels $N_LABELS
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
