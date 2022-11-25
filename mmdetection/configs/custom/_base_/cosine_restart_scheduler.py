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
runner = dict(
    type="EpochBasedRunner",
    max_epochs=50,
)


lr_config = dict(
    policy="CosineRestart",
    warmup="linear",
    warmup_iters=5871,
    warmup_ratio=0.001,
    periods=[19570, 19570, 19570, 19570, 19570],
    restart_weights=[1, 0.8, 0.65, 0.55, 0.5],
    by_epoch=False,
    min_lr=5e-6,
)
