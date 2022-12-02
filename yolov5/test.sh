#!/bin/bash 

for ((i=0; i<=4; i++))
do
    python val.py --weights runs/train/yolov5x_kfold${i}/weights/best.pt --data trash${i}.yaml --img 640 --half --task test --save-json --name yolov5x_kfold${i}_TTA --iou-thres 0.5 --augment
    python val.py --weights runs/train/yolov5x_kfold${i}/weights/best.pt --data trash${i}.yaml --img 640 --half --task test --save-json --name yolov5x_kfold${i} --iou-thres 0.5
done


