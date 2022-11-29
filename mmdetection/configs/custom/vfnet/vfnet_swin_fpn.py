_base_ = [
    "../_base_/best_aug.py",
    "../_base_/runtime.py",
    "../_base_/cosine_restart_scheduler.py",
]

# pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"  # noqa
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth"
# model settings
model = dict(
    type="VFNet",
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
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type="VFNetHead",
        num_classes=10,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type="VarifocalLoss",
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.5),
        loss_bbox_refine=dict(type="GIoULoss", loss_weight=2.0),
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
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

# fp16 = dict(loss_scale="dynamic")
