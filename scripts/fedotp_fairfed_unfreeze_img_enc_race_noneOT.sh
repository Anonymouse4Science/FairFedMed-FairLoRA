#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

# custom config
DATA="DATA/"
MODEL=FedOTPLinearFT
TRAINER=GLP_OT
PRETRAINED=True
OT=None
TOP_PERCENT=0.8
EPS=0.1
THRESH=0.001
MAX_ITER=100
LR=0.001
GAMMA=1
USERS=1
FRAC=0.5
ROUND=50
STEPSIZE=40
NUM_PROMPT=2
# DATASET=$1
CFG=vit_b16_oph  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CTXINIT=False
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
BETA=0.3
INPUT_NO_TRANSFORM=False
# SEED=1
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
ATTRIBUTE_TYPE='race'
UNFREEZE_IMAGE_ENC=True
for DATASET in fairfedmed
do
  for PARTITION in noniid-labeldir100
  do
    for SEED in 1
    do
      DIR=output/${DATASET}_${PARTITION}_beta${BETA}_normalize_unfreeze_image_enc${UNFREEZE_IMAGE_ENC}/${MODEL}_${TRAINER}_${OT}_${TOP_PERCENT}_eps${EPS}_${ATTRIBUTE_TYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
        python federated_main.py \
        --root ${DATA} \
        --model ${MODEL} \
        --seed ${SEED} \
        --num_users ${USERS} \
        --frac ${FRAC} \
        --lr ${LR} \
        --OT ${OT} \
        --top_percent ${TOP_PERCENT} \
        --eps ${EPS} \
        --thresh ${THRESH} \
        --max_iter ${MAX_ITER} \
        --gamma ${GAMMA} \
        --trainer ${TRAINER} \
        --round ${ROUND} \
        --stepsize ${STEPSIZE} \
        --input_no_transform ${INPUT_NO_TRANSFORM} \
        --attribute_type ${ATTRIBUTE_TYPE} \
        --partition ${PARTITION} \
        --beta ${BETA} \
        --n_ctx ${NCTX} \
        --num_prompt ${NUM_PROMPT} \
        --unfreeze_image_encoder ${UNFREEZE_IMAGE_ENC} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GLP_OT/${CFG}.yaml \
        --output-dir ${DIR}
      fi
    done
  done
done
