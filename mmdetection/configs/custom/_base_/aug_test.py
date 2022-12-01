# dataset settings
dataset_type = "CocoDataset"
data_root = "../dataset/"

TRAIN_DATA = "../stratify_dataset/train_fold_0.json"
VAL_DATA = "../stratify_dataset/val_fold_0.json"
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375], to_rgb=True
)


albu_train_transforms = [
    dict(type="RandomRotate90", p=1.0),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="HueSaturationValue",
                hue_shift_limit=10,
                sat_shift_limit=35,
                val_shift_limit=25,
            ),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=0.25,
                contrast_limit=0.25,
            ),
            dict(type="RGBShift", r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.4,
    ),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=TRAIN_DATA,  # train json 파일 경로
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline,
    ),
    persistent_workers=True,
    val=dict(
        type=dataset_type,
        ann_file=VAL_DATA,  # valid json 파일 경로
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=f"{data_root}test.json",  # test json 파일 경로
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50", classwise=True)
# evaluation=dict(save_best='mAP')
