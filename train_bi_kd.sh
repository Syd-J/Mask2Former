#!/bin/bash

echo "activating environment"
source /home/sid/miniconda3/bin/activate
conda activate bi-kd

export DETECTRON2_DATASETS=/home/sid/ade20k

echo "starting train_bi_kd.py"
python train_bi_kd.py \
  --t_config configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml \
  --s_config configs/ade20k/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_160k.yaml \
  MODEL.WEIGHTS /home/sid/Mask2Former/pretrained/model_final_R50.pkl \
  MODEL.WEIGHTS /home/sid/Mask2Former/pretrained/model_final_Swin_T.pkl

conda deactivate
