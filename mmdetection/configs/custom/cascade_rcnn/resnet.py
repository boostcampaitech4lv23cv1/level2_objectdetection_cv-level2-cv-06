_base_ = [
    "../../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/dataset.py",
    "../_base_/runtime.py",
    "../_base_/base_scheduler.py",
]

model = dict(
    type="CascadeRCNN",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
)
