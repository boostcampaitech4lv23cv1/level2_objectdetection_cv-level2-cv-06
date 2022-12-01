# # python tools/train.py configs/custom/cascade_rcnn/pvt_b1.py --work-dir work_dirs/cascade_rcnn_pvt_b1 --seed 42

<<<<<<< HEAD
# python tools/train.py configs/custom/atss/resnet50.py \
# --wandb_nm atss_res50_batch_64 --wandb_tag atss resnet


# python tools/train.py configs/custom/cascade_rcnn/eff_b0.py

python tools/train.py \
    configs/custom/cascade_rcnn/cascade_r50_fpn_score_thr_dot03.py \
    --work-dir ../work_dirs/cascade_r50_fpn_batch_8_score_thr_dot03 \
    --seed 42 \
    --wandb_nm cascade_r50_fpn_batch_8_score_thr_dot03 \
    --wandb_tag cascade_rcnn \
    --batch_size 8 \
    --epochs 20

# python tools/train.py \
#     configs/custom/cascade_rcnn/cascade_r50_fpn_score_thr_dot00.py \
#     --work-dir ../work_dirs/cascade_r50_fpn_batch_8_score_thr_dot00 \
#     --seed 42 \
#     --wandb_nm cascade_r50_fpn_batch_8_score_thr_dot00 \
#     --wandb_tag cascade_rcnn \
#     --batch_size 8 \
#     --epochs 20

# python tools/train.py \
#     configs/custom/cascade_rcnn/cascade_r50_fpn_score_thr_dot01.py \
#     --work-dir ../work_dirs/cascade_r50_fpn_batch_8_score_thr_dot01 \
#     --seed 42 \
#     --wandb_nm cascade_r50_fpn_batch_8_score_thr_dot01 \
#     --wandb_tag cascade_rcnn \
#     --batch_size 8 \
#     --epochs 20
=======
python tools/train.py configs/custom/cascade_rcnn/swinv2_large.py \
--wandb_nm cascade_swinv2_large_augmentation --wandb_tag cascade augmentation swinv2_large



>>>>>>> DEV/main
