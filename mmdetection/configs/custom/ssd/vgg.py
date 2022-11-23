_base_ = [
    "../../_base_/models/ssd300.py",
    "../_base_/dataset.py",
    "../_base_/runtime.py",
    "../_base_/base_scheduler.py",
]

model = dict(
    bbox_head=dict(
        type="SSDHead",
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=10,
        anchor_generator=dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
    ),
)
