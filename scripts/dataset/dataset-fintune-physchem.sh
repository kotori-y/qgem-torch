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

CONFIG_PATH='./config.json'
DATASET_ROOT='./data/downstream/physchem-sample'
DATASET_BASE_PATH='./data/downstream/physchem-sample'

SEED=2024

ENDPOINTS='Density Vapor_Pressure Melting_Point Boiling_Point Flash_Point Decomposition Surface_Tension Drug_Half_Life Viscosity LogS Refractive_Index LogP Solubility'

echo "run as dataset mode"
echo "=================="
python finetune_regr.py \
  --config-path $CONFIG_PATH \
  --dataset-root $DATASET_ROOT \
  --dataset-base-path $DATASET_BASE_PATH \
  --endpoints $ENDPOINTS \
  --remove-hs \
  --dataset
