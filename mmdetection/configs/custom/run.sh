python tools/train.py \
    configs/custom/cascade_rcnn/cascade_r50_fpn.py \
    --work-dir ../work_dirs/cascade_r50_fpn_batch_32 \
    --seed 42 \
    --wandb_nm cascade_r50_fpn_batch_32 \
    --wandb_tag cascade_rcnn 


# python tools/train.py configs/custom/cascade_rcnn/pvt_b1.py --work-dir work_dirs/cascade_rcnn_pvt_b1 --seed 42

python tools/train.py configs/custom/cascade_rcnn/pvt_b1.py \
--wandb_nm cascade_rcnn_pvt_b1 --wandb_tag pvt cascade_rcnn

