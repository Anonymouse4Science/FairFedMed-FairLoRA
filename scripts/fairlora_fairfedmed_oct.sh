#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# custom config
DATA="DATA/"
MODEL=FedOTPLoRA
TRAINER=GLP_OT_SVLoRA
PRETRAINED=True
OT=None
TOP_PERCENT=0.8
EPS=0.1
THRESH=0.001
MAX_ITER=100
LR=0.001
GAMMA=0.1
USERS=3
FRAC=0.8
ROUND=50
STEPSIZE=200  # 2 * ROUND * 0.5 * max(USERS * FRAC, 1)
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
# params for SVLoRA
ATTRIBUTE_TYPE='language'
MODALITY_TYPE='oct_bscans'
DIM_PER_3D_SLICE=8
UNFREEZE_IMAGE_ENC=True
LoRA_RANK=12
LoRA_ALPHA=2
LoRA_TYPE=FairLoRA
for DATASET in fairfedmed
do
  for PARTITION in noniid-labeldir100
  do
    for SEED in 1
    do
      DIR=output/FairLoRA_${CFG}_oct_ema/${DATASET}_${MODALITY_TYPE}_${PARTITION}_beta${BETA}_${LoRA_TYPE}_rank${LoRA_RANK}_alpha${LoRA_ALPHA}_itv4_slice${DIM_PER_3D_SLICE}/${MODEL}_${TRAINER}_${OT}_${TOP_PERCENT}_eps${EPS}_${ATTRIBUTE_TYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
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
        --modality_type ${MODALITY_TYPE} \
        --dim_per_3d_slice ${DIM_PER_3D_SLICE} \
        --partition ${PARTITION} \
        --beta ${BETA} \
        --n_ctx ${NCTX} \
        --num_prompt ${NUM_PROMPT} \
        --unfreeze_image_encoder ${UNFREEZE_IMAGE_ENC} \
        --lora_rank ${LoRA_RANK} \
        --lora_alpha ${LoRA_ALPHA} \
        --lora_type ${LoRA_TYPE} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GLP_OT/${CFG}.yaml \
        --output-dir ${DIR}
      fi
    done
  done
done