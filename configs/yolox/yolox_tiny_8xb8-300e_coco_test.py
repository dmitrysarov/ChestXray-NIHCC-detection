_base_ = ["../_base_/schedules/schedule_1x.py", "../_base_/default_runtime.py", "./yolox_tta.py"]

dataset_type = "CocoDataset"
data_root = "../test_data/"
backend_args = None

# model settings

model = dict(
    type="YOLOX",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[122.48, 122.48, 122.48],
        std=[64.48, 64.48, 64.48],
        batch_augments=[],
    ),
    backbone=dict(
        type="CSPDarknet",
        deepen_factor=0.33,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=7,  # 7 - because of test mockdataset
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
        loss_obj=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0),
        loss_l1=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)
# img_scale = (640, 640)  # width, height

train_pipeline = [
    # dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type="RandomAffine",
    #     scaling_ratio_range=(0.5, 1.5),
    #     # img_scale is (width, height)
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
    # ),
    # dict(type="YOLOXHSVRandomAug"),
    # dict(type="RandomFlip", prob=0.5),
    # Resize and Pad are for the last 15 epochs when Mosaic and
    # RandomAffine are closed by YOLOXModeSwitchHook.
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(416, 416), keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(122.0, 122.0, 122.0))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(classes=[]),
    # ann_file="BBox_List_2017_train.json",
    ann_file="BBox_List_2017_val.json",
    data_prefix=dict(img="../data/images1000/"),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    backend_args=backend_args,
    pipeline=train_pipeline,
)
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)
test_pipeline = [
    # dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(416, 416), keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(122.0, 122.0, 122.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=[]),
        data_root=data_root,
        ann_file="BBox_List_2017_val.json",
        data_prefix=dict(img="../data/images1000/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "BBox_List_2017_val.json",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator
max_epochs = 300
num_last_epochs = 15
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

base_lr = 0.001
optim_wrapper = dict(
    type="OptimWrapper",
    # optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    optimizer=dict(type="AdamW", lr=base_lr),
    # paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# param_scheduler = [
#     dict(
#         # use quadratic formula to warm up 5 epochs
#         # and lr is updated by iteration
#         # TODO: fix default scope in get function
#         type="mmdet.QuadraticWarmupLR",
#         by_epoch=True,
#         begin=0,
#         end=5,
#         convert_to_iter_based=True,
#     ),
#     dict(
#         # use cosine lr from 5 to 285 epoch
#         type="CosineAnnealingLR",
#         eta_min=base_lr * 0.05,
#         begin=5,
#         T_max=max_epochs - num_last_epochs,
#         end=max_epochs - num_last_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
#     dict(
#         # use fixed lr during last 15 epochs
#         type="ConstantLR",
#         by_epoch=True,
#         factor=1,
#         begin=max_epochs - num_last_epochs,
#         end=max_epochs,
#     ),
# ]

default_hooks = dict(checkpoint=dict(interval=interval, max_keep_ckpts=3))  # only keep latest 3 checkpoints

custom_hooks = [
    # dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    # dict(type="SyncNormHook", priority=48),
    # dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0001, update_buffers=True, priority=49),
    dict(type="MLflowHook")
]
auto_scale_lr = dict(base_batch_size=16)
