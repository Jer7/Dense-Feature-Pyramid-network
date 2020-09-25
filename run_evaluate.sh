#!/bin/bash

CS_PATH='./dataset/Cartoon_sketches/Dog/'
# CS_PATH='./dataset/LIP/TrainVal'
BS=12
GPU_IDS='0'
INPUT_SIZE='384,384'
# SNAPSHOT_FROM='./Trained_Models/Original_params+IoU/LIP_epoch_4500.pth'
SNAPSHOT_FROM='./snapshots/'
# SNAPSHOT_FROM='/home/jeromewan/SJTU_Thesis/Non_local_CE2P/Trained_Models/1000_images/UResnet_v3_ASPP/HD/final_concat_conv3x3_afetr_upsampling/LIP_epoch_495.pth'
#DATASET='./dataset/LIP/TrainVal_images/TrainVal_images'
DATASET='val'
NUM_CLASSES=8
# NUM_CLASSES=20
SAVE_PATH_DIR='./dataset/Cartoon_sketches/Dog/save_preds/'

python evaluate_save_preds.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-path ${SAVE_PATH_DIR}