#!/bin/bash 

for ((i=0; i<=4; i++))
do
    python train.py --batch 64 --data trash${i}.yaml --weights yolov5x.pt --device 0 --name yolov5x_kfold${i} --img 640 \
    --epochs 300 --multi-scale --seed 42 --sync-bn --cache --patience 10
done