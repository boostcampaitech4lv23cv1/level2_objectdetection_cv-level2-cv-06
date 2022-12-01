python tools/train.py \
    configs/custom/vfnet/vfnet_swinv2_large_fpn.py \
    --work-dir ./work_dirs/vfnet_swinv2_large_fpn \
    --seed 42 \
    --wandb_nm VFnet_swinv2_large_fpn \
    --wandb_tag vfnet swin fpn \
    --batch_size 2 \
    --epochs 50
