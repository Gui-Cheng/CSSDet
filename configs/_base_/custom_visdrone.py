_base_ = [
    'schedules/schedule_1x.py', 'default_runtime.py'
]

img_scale = (1333, 800) ### (w,h)

# img_scale = (800,640)  ### (w,h)
# img_scale = (1200,675)  ### (w,h)
# dataset settings
data_root = '../datasets/'
dataset_type = 'CocoDataset'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')


file_client_args = dict(backend='disk')



####数据增强
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # dict(
    #     type='RandomChoiceResize',
    #     # scales=[(800,640),(1200,960),(1200,675),(1024,540)],
    #     scales=[(800,640),(1024,540)],
    #     keep_ratio=True,
    #     backend='pillow'),
    # dict(
    #     type='RandomResize',
    #     scale=img_scale,
    #     ratio_range=(0.8, 1.25),
    #     keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=img_scale,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
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
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= 'VisDrone2019/annotations/VisDrone2019-DET_train_coco.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img='VisDrone2019/VisDrone2019-DET-train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='VisDrone2019/annotations/VisDrone2019-DET_val_coco.json',
        data_prefix=dict(img='VisDrone2019/VisDrone2019-DET-val/images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VisDrone2019/annotations/VisDrone2019-DET_val_coco.json',
    metric='bbox',
    format_only=False,
    classwise=True)
test_evaluator = val_evaluator


# training settings
max_epochs = 200
num_last_epochs = 15
interval = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)




# # optimizer
# # default 8 gpu
# base_lr = 0.01
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
#         nesterov=True),
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# # lr steps at [0.9, 0.95] of the maximum iterations
# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.067, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[180, 190],
#         gamma=0.1)
# ]


### yolox学习率策略
# optimizer
# default 8 gpu
base_lr = 0.02
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=2,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]



# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP'# only keep latest 3 checkpoints
    ),
    visualization= dict(type='DetVisualizationHook', 
    interval=2,  # 保存间隔改成 1，每张图片都保存
    draw= False) # 重要！开启绘制和保存功能，默认不开启

    )


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]### 预测图像存储到TensorBoard

# # ### 测试时数据增强
# tta_model = dict(
#     type='DetTTAModel',
#     tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

# img_scales = [(800, 640), (1200, 960),(1024,675)]
# # img_scales = [(1024,1024),(1200,1200)]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
#     dict(
#         type='TestTimeAug',
#         transforms=[[
#             dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
#         ], [
#             dict(type='RandomFlip', prob=1.),
#             dict(type='RandomFlip', prob=0.)
#         ], [dict(type='LoadAnnotations', with_bbox=True)],
#                     [
#                         dict(
#                             type='PackDetInputs',
#                             meta_keys=('img_id', 'img_path', 'ori_shape',
#                                        'img_shape', 'scale_factor', 'flip',
#                                        'flip_direction'))
#                     ]])
# ]
