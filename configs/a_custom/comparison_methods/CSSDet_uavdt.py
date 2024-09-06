_base_ = [
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py',
    '../yolox_tta.py'
]
#### UAVDT (1024*540)
#### VisDrone(1200,675)
img_scale = (800,640)  # width, height
# img_scale = (640,640)  # width, height
# model settings
model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        # batch_augments=[
        #     dict(
        #         type='BatchSyncRandomResize',
        #         random_size_range=(480, 800),
        #         size_divisor=32,
        #         interval=10)
        # ]
        ),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(1,2, 3, 4),###(4,8,16,32) P2, P3,P4,P5
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN_v6',
        in_channels=[128,256, 512, 1024],###(4,8,16,32) P2, P3,P4,P5
        out_channels=256,
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=(4,8, 16, 32),###(4,8,16,32) P2, P3,P4,P5
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = '../datasets/'
dataset_type = 'CocoDataset'
classes = ("car", "truck", "bus")

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= 'UAVDT/UAVDT_annotations/UAVDT_train_coco.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img='UAVDT/UAVDT_images/UAVDT_train/'),
        pipeline=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='UAVDT/UAVDT_annotations/UAVDT_test_coco.json',
        data_prefix=dict(img='UAVDT/UAVDT_images/UAVDT_test/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=6)


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'UAVDT/UAVDT_annotations/UAVDT_test_coco.json',
    metric='bbox',
    classwise=True)
test_evaluator = val_evaluator

# training settings
max_epochs = 50
num_last_epochs = 10
interval = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=interval,dynamic_intervals=[(max_epochs - num_last_epochs, 1)])

# optimizer
# default 8 gpu
base_lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# # learning rate
# param_scheduler = [
#     dict(
#         # use quadratic formula to warm up 5 epochs
#         # and lr is updated by iteration
#         # TODO: fix default scope in get function
#         type='mmdet.QuadraticWarmupLR',
#         by_epoch=True,
#         begin=0,
#         end=2,
#         convert_to_iter_based=True),
#     dict(
#         # use cosine lr from 5 to 285 epoch
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.05,
#         begin=2,
#         T_max=max_epochs - num_last_epochs,
#         end=max_epochs - num_last_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         # use fixed lr during last 15 epochs
#         type='ConstantLR',
#         by_epoch=True,
#         factor=1,
#         begin=max_epochs - num_last_epochs,
#         end=max_epochs,
#     )
# ]




# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[40, 45],
        gamma=0.1)
]



default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3,  # only keep latest 3 checkpoints
        save_best='coco/bbox_mAP'
    ),
    visualization= dict(type='DetVisualizationHook', 
    interval=2,  # 保存间隔改成 1，每张图片都保存
    draw= False, # 重要！开启绘制和保存功能，默认不开启
    draw_gt=False)
    )

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]### 预测图像存储到TensorBoard


custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]


