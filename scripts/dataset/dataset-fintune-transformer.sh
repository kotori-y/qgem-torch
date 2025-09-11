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

CONFIG_PATH='./configs/config.json'
DATASET_BASE_PATH='./data/downstream/qc_demo'
# DATASET_BASE_PATH='./data/downstream/toxicity_nova_v5_2'
# DATASET_BASE_PATH='./data/downstream/physchem_nova_v2'

SEED=2024

# ENDPOINTS='Density Vapor_Pressure Melting_Point Boiling_Point Flash_Point Decomposition Surface_Tension Drug_Half_Life Viscosity LogS Refractive_Index LogP Solubility'
# ENDPOINTS="Decomposition Density Heat_of_Combustion Melting_Point"

# ENDPOINTS="Cat_Intravenous_LD50 Chicken_Oral_LD50 Guineapig_Oral_LD50 Mouse_Intravenous_LD50 Mouse_Subcutaneous_LD50 Rabbit_Oral_LD50 Rat_Intravenous_LD50 Rat_Subcutaneous_LD50 Cat_Oral_LD50 Dog_Intravenous_LD50 Mouse_Intraperitoneal_LD50 Mouse_Oral_LD50 Rabbit_Intravenous_LD50 Rat_Intraperitoneal_LD50 Rat_Oral_LD50"
# ENDPOINTS="Boiling_Point Density Flash_Point LogP LogS Melting_Point Refractive_Index Surface_Tension Vapor_Pressure"
ENDPOINTS="alphaOpt density eGapOpt eLumoOpt espAvg espcMin espVarNeg hCorre mpi npaMin sa saPos uCorre zpe cm5Max e0K eHOMO eOpt espAvgNeg espMax espVarPos H mu nu saNeg S U cm5Min e eHomoOpt ese espAvgPos espMin gCorre hirshfeldMax muOpt pi saNonpolar theta volumeIMT cv eGap eLUMO eseOpt espcMax espVar G hirshfeldMin npaMax productOfSigmaSquareAndNu saPolar thetaOpt volumeMC"

PREPROCESS_ENDPOINTS="kotori"

N_NODES=$1

echo "run as dataset mode"
echo "=================="
mpirun -n $N_NODES python finetune_transformer.py \
  --config-path $CONFIG_PATH \
  --dataset-base-path $DATASET_BASE_PATH \
  --endpoints $ENDPOINTS \
  --preprocess-endpoints $PREPROCESS_ENDPOINTS \
  --remove-hs \
  --dataset \
  --use_mpi
