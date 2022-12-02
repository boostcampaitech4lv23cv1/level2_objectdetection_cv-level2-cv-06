_base_ = [
    "../_base_/best_aug.py",
    "../_base_/runtime.py",
    "../_base_/base_scheduler.py",
]


pretrained = "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth"  # noqa

model = dict(
    type="ATSS",
    backbone=dict(
        # _delete_=True,
        type="SwinTransformerV2",
        pretrain_img_size=192,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(
        type="FPN",
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        type="ATSSHead",
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.00,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
    ),
)
fp16 = dict(loss_scale="dynamic")