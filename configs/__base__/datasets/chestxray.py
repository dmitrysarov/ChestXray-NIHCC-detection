# pipelines not described here because they part of main config file, e.g. yolox/yolox_tiny_8xb8-300e_coco.py
dataset_type = "ChessXray"

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        ann_file="/kaggle/working/BBox_List_2017_train.csv",
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="/kaggle/working/BBox_List_2017_val.csv",
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file="/kaggle/working/BBox_List_2017_val.csv",
    metric="bbox",
    format_only=False,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="/kaggle/working/BBox_List_2017_test.csv",
        test_mode=True,
    ),
)

test_evaluator = dict(
    type="CocoMetric",
    metric="bbox",
    format_only=False,
    ann_file="/kaggle/working/BBox_List_2017_test.csv",
    outfile_prefix="./kaggle/working/test",
)
