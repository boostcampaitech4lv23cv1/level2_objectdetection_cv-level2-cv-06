# dataset settings
dataset_type = "CocoDataset"
data_root = "/opt/ml/dataset/"
classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
img_norm_cfg = dict(
    mean=[123.65, 117.397, 110.075], std=[60.266, 59.257, 61.373], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="MixUp", img_scale=(512, 512)),
    # dict(type='Resize', img_scale=(512,512), multiscale_mode='range', keep_ratio=True), #이미지 사이즈 랜덤 선택
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "train_split.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=train_dataset,
    persistent_workers=True,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "valid_split.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
"b"
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
# evaluation=dict(save_best='mAP')
