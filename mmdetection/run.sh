# python tools/train.py configs/custom/cascade_rcnn/pvt_b1.py --work-dir work_dirs/cascade_rcnn_pvt_b1 --seed 42

# python tools/train.py configs/custom/atss/resnet50.py \
# --wandb_nm atss_res50_batch_64 --wandb_tag atss resnet


# python tools/train.py configs/custom/cascade_rcnn/eff_b0.py

# python tools/train.py \
#     configs/custom/vfnet/vfnet_swin_fpn.py \
#     --work-dir ../work_dirs/vfnet_swin_fpn_batch_32 \
#     --seed 42 \
#     --wandb_nm vfnet_swin_fpn_batch_32 \
#     --wandb_tag vfnet swin fpn \
#     --batch_size 32 \
#     --epochs 80

# python tools/train.py \
#     configs/custom/vfnet/vfnet_swin_fpn.py \
#     --work-dir ../work_dirs/vfnet_swin_fpn_batch_16 \
#     --seed 42 \
#     --wandb_nm VFNet_swin_fpn_batch_16 \
#     --wandb_tag vfnet swin fpn \
#     --batch_size 16 \
#     --epochs 40

# python tools/train.py configs/custom/cascade_rcnn/swinv2_large.py \
# --wandb_nm cascade_swinv2_large_augmentation --wandb_tag cascade augmentation swinv2_large

python tools/train.py \
    configs/custom/vfnet/vfnet_swin_fpn.py \
    --work-dir ../work_dirs/vfnet_swin_fpn_batch_2 \
    --seed 42 \
    --wandb_nm VFnet_swin_fpn_batch_2 \
    --wandb_tag vfnet swin fpn \
    --batch_size 2 \
    --epochs 50




