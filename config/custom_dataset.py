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
    # dict(
    #   type="Resize", img_scale=(512, 1024), multiscale_mode="range", keep_ratio=True
    # ),  # 이미지 사이즈 랜덤 선택
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


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="/opt/ml/stratify_dataset/train_fold_0.json",  # train json 파일 경로
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline,
    ),
    persistent_workers=True,
    val=dict(
        type=dataset_type,
        ann_file="/opt/ml/stratify_dataset/val_fold_0.json",  # valid json 파일 경로
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",  # test json 파일 경로
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
# evaluation=dict(save_best='mAP')
