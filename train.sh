# pwd opt/ml
# custom_conifg => mmdetection/configs/custom에서 수정
# sh run train.sh
cd mmdetection
python tools/train.py configs/custom/cascade_swin.py --work-dir log