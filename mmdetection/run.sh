# python tools/train.py configs/custom/cascade_rcnn/pvt_b1.py --work-dir work_dirs/cascade_rcnn_pvt_b1 --seed 42

python tools/train.py configs/custom/atss/resnet50.py \
--wandb_nm atss_res50_batch_64 --wandb_tag atss resnet


python tools/train.py configs/custom/cascade_rcnn/eff_b0.py
