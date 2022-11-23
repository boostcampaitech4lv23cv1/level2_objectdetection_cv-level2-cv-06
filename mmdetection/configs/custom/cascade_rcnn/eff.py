_base_ = [
    "../../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/dataset.py",
    "../_base_/runtime.py",
    "../_base_/base_scheduler.py",
]

custom_imports = dict(imports=["mmcls.models"], allow_failed_imports=False)
model = dict(
    type="CascadeRCNN",
    backbone=dict(
        _delete_=True,
        type="mmcls.TIMMBackbone",
        model_name="efficientnet_b0",
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4),
    ),
    neck=dict(type="FPN", in_channels=[24, 40, 112, 320], out_channels=256, num_outs=5),
)
