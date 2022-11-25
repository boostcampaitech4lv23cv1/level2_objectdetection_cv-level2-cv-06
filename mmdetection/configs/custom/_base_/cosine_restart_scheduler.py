lr_config = dict(
    policy="CosineRestart",
    warmup="linear",
    warmup_iters=980,
    warmup_ratio=0.001,
    periods=[3266, 3266, 3266],
    restart_weights=[1, 0.75, 0.5],
    by_epoch=False,
    min_lr=1e-6,
)
