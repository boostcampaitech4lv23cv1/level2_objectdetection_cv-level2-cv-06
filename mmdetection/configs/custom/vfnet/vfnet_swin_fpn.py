_base_ = [
    "../_base_/dataset.py",
    "../_base_/base_scheduler.py",
    "../_base_/runtime.py",
]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"  # noqa
# model settings
model = dict(
    type="VFNet",
    backbone=dict(
        type="SwinTransformer",
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(
        type="FPN",
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[128, 256, 512, 1024],
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
