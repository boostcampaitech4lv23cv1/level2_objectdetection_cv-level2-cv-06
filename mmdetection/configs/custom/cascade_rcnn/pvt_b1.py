_base_ = [
    "../../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/dataset.py",
    "../_base_/runtime.py",
    "../_base_/base_scheduler.py",
]

model = dict(
    type="CascadeRCNN",
    backbone=dict(
        _delete_=True,
        type="PyramidVisionTransformerV2",
        embed_dims=64,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(
            checkpoint="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth"
        ),
    ),
    neck=dict(type="FPN", in_channels=[64, 128, 320, 512]),
)
