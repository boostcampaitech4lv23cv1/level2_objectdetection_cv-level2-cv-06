#!/bin/bash 

for ((i=0; i<=4; i++))
do
    python runs/convert2csv.py --file yolov5x_kfold${i}_TTA
    python runs/convert2csv.py --file yolov5x_kfold${i}
done