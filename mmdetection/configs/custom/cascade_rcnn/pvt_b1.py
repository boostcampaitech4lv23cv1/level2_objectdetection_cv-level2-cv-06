_base_ = [
    "../../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/dataset.py",
    "../_base_/runtime.py",
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

optimizer = dict(
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 15],
)
runner = dict(
    type="EpochBasedRunner",
    max_epochs=20,
)