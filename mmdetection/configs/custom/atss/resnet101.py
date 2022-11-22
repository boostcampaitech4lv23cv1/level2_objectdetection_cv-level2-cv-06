_base_ = [
    "./resnet50.py",
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
)
