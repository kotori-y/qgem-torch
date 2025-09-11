#!/bin/bash

ENCODER_EVAL_FROM='./outs/checkpoints/pretrain/zinc22-5-gin/checkpoint_10.pt'
# ENCODER_EVAL_FROM='kotori'


select PRETRAINED_SIZE in "10" "5" "2" "1"; do
  ENCODER_EVAL_FROM="./outs/checkpoints/pretrain/zinc22-${PRETRAINED_SIZE}-gin/checkpoint_10.pt"
  break;
done
echo "use ${ENCODER_EVAL_FROM}"
OUT_DIR="${PRETRAINED_SIZE}0M"

# lrs_list="1e-3,1e-3 1e-3,4e-3 4e-3,4e-3 4e-4,4e-3 1e-4,1e-3 1e-4,1e-4"
# drop_list="0.0 0.1 0.2 0.3 0.4"

lrs_list="1e-4,1e-3"
drop_list="0.4"

batch_size_list="128"

# seed_list="2025 1996 805 621 111 222 333 444 555 666"
# seed_list="2025 1996 805 621 777 555"
# seed_list="2025 777 912 923 555"
seed_list="2025"

for BATCH_SIZE in $batch_size_list; do

  for lrs in $lrs_list; do
		IFS=, read -r -a array <<< "$lrs"
		lr=${array[0]}
		head_lr=${array[1]}
    echo "${lr}@${head_lr}"

    for dropout_rate in $drop_list; do
      for seed in $seed_list; do
        sh finetune-transformer.sh 0 $ENCODER_EVAL_FROM $OUT_DIR $lr $head_lr $dropout_rate $BATCH_SIZE $seed
      done
    done
  done
done
