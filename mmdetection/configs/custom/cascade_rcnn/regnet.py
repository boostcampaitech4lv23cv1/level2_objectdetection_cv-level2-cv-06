_base_ = [
    "../../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/custom_dataset.py",
    "../_base_/runtime.py",
    "../_base_/base_scheduler.py",
]


model = dict(
    type="CascadeRCNN",
    backbone=dict(
        type="RegNet",
        arch="regnetx_12gf",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://regnetx_12gf"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[224, 448, 896, 2240],
        out_channels=256,
        num_outs=5,
    ),
)
