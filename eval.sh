#!/bin/bash

echo "activating environment"
source ~/miniconda3/bin/activate
conda activate DL

export DETECTRON2_DATASETS=/home/sid/ade20k

echo "starting train_net.py"
python train_net.py \
  --config-file configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml \
  --eval-only MODEL.WEIGHTS /home/sid/Mask2Former/pretrained/model_final_R50.pkl

conda deactivate
