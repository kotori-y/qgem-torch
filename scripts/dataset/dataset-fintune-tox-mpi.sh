#!/bin/bash

cd ../..

source /fs1/software/modules/bashrc
module load loginnode
module load CUDA/11.8
module load cudnn/8.4.1-cuda11.x
export PYTHONUNBUFFERED=1

# root_path="../../../.."
# export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

CONFIG_PATH='./config/config.json'
DATASET_ROOT='./data/downstream/toxicity'
DATASET_BASE_PATH='./data/downstream/toxicity'

SEED=2024

N_NODES=$1

ENDPOINTS="Cat_Intravenous_LD50 Cat_Oral_LD50 Chicken_Oral_LD50 Dog_Oral_LD50 Duck_Oral_LD50 Guineapig_Oral_LD50 Mouse_Intramuscular_LD50 Mouse_Intraperitoneal_LD50 Mouse_Intravenous_LD50 Mouse_Oral_LD50 Mouse_Subcutaneous_LD50 Rabbit_Intravenous_LD50 Rabbit_Oral_LD50 Rat_Inhalation_LC50 Rat_Intraperitoneal_LD50 Rat_Intravenous_LD50 Rat_Oral_LD50 Rat_Skin_LD50 Rat_Subcutaneous_LD50"

echo "run as dataset mode"
echo "=================="
mpirun -n $N_NODES python finetune_regr.py \
  --config-path $CONFIG_PATH \
  --dataset-root $DATASET_ROOT \
  --dataset-base-path $DATASET_BASE_PATH \
  --endpoints $ENDPOINTS \
  --remove-hs \
  --useMPI \
  --dataset
