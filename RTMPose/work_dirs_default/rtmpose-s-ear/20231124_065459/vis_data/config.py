NUM_KEYPOINTS = 21
auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
base_lr = 0.004
codec = dict(
    input_size=(
        256,
        256,
    ),
    normalize=False,
    sigma=(
        12,
        12,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=300,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=60,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                shift_factor=0.0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        256,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        12,
                        12,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = 'ear_pose/Ear210_Keypoint_Dataset_coco/'
dataset_info = dict(
    classes='ear',
    dataset_name='Ear210_Keypoint_Dataset_coco',
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    keypoint_info=dict({
        0:
        dict(color=[
            101,
            205,
            228,
        ], id=0, name='肾上腺', swap='', type=''),
        1:
        dict(color=[
            240,
            128,
            128,
        ], id=1, name='耳尖', swap='', type=''),
        10:
        dict(color=[
            128,
            0,
            128,
        ], id=10, name='膀胱', swap='', type=''),
        11:
        dict(color=[
            74,
            181,
            57,
        ], id=11, name='脾', swap='', type=''),
        12:
        dict(color=[
            165,
            42,
            42,
        ], id=12, name='角窝中', swap='', type=''),
        13:
        dict(color=[
            128,
            128,
            0,
        ], id=13, name='神门', swap='', type=''),
        14:
        dict(color=[
            255,
            0,
            0,
        ], id=14, name='肾', swap='', type=''),
        15:
        dict(color=[
            34,
            139,
            34,
        ], id=15, name='耳门', swap='', type=''),
        16:
        dict(color=[
            255,
            129,
            0,
        ], id=16, name='听宫', swap='', type=''),
        17:
        dict(color=[
            70,
            130,
            180,
        ], id=17, name='听会', swap='', type=''),
        18:
        dict(color=[
            63,
            103,
            165,
        ], id=18, name='肩', swap='', type=''),
        19:
        dict(color=[
            66,
            77,
            229,
        ], id=19, name='扁桃体', swap='', type=''),
        2:
        dict(color=[
            154,
            205,
            50,
        ], id=2, name='胃', swap='', type=''),
        20:
        dict(color=[
            255,
            105,
            180,
        ], id=20, name='腰骶椎', swap='', type=''),
        3:
        dict(color=[
            34,
            139,
            34,
        ], id=3, name='眼', swap='', type=''),
        4:
        dict(color=[
            139,
            0,
            0,
        ], id=4, name='口', swap='', type=''),
        5:
        dict(color=[
            255,
            165,
            0,
        ], id=5, name='肝', swap='', type=''),
        6:
        dict(color=[
            255,
            0,
            255,
        ], id=6, name='对屏尖', swap='', type=''),
        7:
        dict(color=[
            255,
            255,
            0,
        ], id=7, name='心', swap='', type=''),
        8:
        dict(color=[
            29,
            123,
            243,
        ], id=8, name='肺', swap='', type=''),
        9:
        dict(color=[
            0,
            255,
            255,
        ], id=9, name='肺2', swap='', type='')
    }),
    paper_info=dict(
        author='Tongji Zihao',
        container='OpenMMLab',
        homepage='https://space.bilibili.com/1900783',
        title='Triangle Keypoints Detection',
        year='2023'),
    sigmas=[
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
    ],
    skeleton_info=dict({
        0:
        dict(color=[
            100,
            150,
            200,
        ], id=0, link=(
            '眼',
            '扁桃体',
        )),
        1:
        dict(color=[
            200,
            100,
            150,
        ], id=1, link=(
            '耳门',
            '听宫',
        )),
        2:
        dict(color=[
            150,
            120,
            100,
        ], id=2, link=(
            '听宫',
            '听会',
        )),
        3:
        dict(color=[
            66,
            77,
            229,
        ], id=3, link=(
            '耳门',
            '听会',
        ))
    }))
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        _scope_='mmpose',
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        _scope_='mmpose',
        interval=10,
        max_keep_ckpts=2,
        rule='greater',
        save_best='PCK',
        type='CheckpointHook'),
    logger=dict(_scope_='mmpose', interval=1, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpose', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpose', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpose', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpose', enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmpose',
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.33,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.5),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                256,
                256,
            ),
            normalize=False,
            sigma=(
                12,
                12,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=512,
        in_featuremap_size=(
            8,
            8,
        ),
        input_size=(
            256,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=21,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=20, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
stage2_num_epochs = 0
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='val_coco.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='ear_pose/Ear210_Keypoint_Dataset_coco/',
        metainfo=dict(
            classes='ear',
            dataset_name='Ear210_Keypoint_Dataset_coco',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        101,
                        205,
                        228,
                    ],
                    id=0,
                    name='肾上腺',
                    swap='',
                    type=''),
                1:
                dict(
                    color=[
                        240,
                        128,
                        128,
                    ], id=1, name='耳尖', swap='', type=''),
                10:
                dict(
                    color=[
                        128,
                        0,
                        128,
                    ], id=10, name='膀胱', swap='', type=''),
                11:
                dict(color=[
                    74,
                    181,
                    57,
                ], id=11, name='脾', swap='', type=''),
                12:
                dict(
                    color=[
                        165,
                        42,
                        42,
                    ], id=12, name='角窝中', swap='', type=''),
                13:
                dict(
                    color=[
                        128,
                        128,
                        0,
                    ], id=13, name='神门', swap='', type=''),
                14:
                dict(color=[
                    255,
                    0,
                    0,
                ], id=14, name='肾', swap='', type=''),
                15:
                dict(
                    color=[
                        34,
                        139,
                        34,
                    ], id=15, name='耳门', swap='', type=''),
                16:
                dict(
                    color=[
                        255,
                        129,
                        0,
                    ], id=16, name='听宫', swap='', type=''),
                17:
                dict(
                    color=[
                        70,
                        130,
                        180,
                    ], id=17, name='听会', swap='', type=''),
                18:
                dict(
                    color=[
                        63,
                        103,
                        165,
                    ], id=18, name='肩', swap='', type=''),
                19:
                dict(
                    color=[
                        66,
                        77,
                        229,
                    ], id=19, name='扁桃体', swap='', type=''),
                2:
                dict(color=[
                    154,
                    205,
                    50,
                ], id=2, name='胃', swap='', type=''),
                20:
                dict(
                    color=[
                        255,
                        105,
                        180,
                    ],
                    id=20,
                    name='腰骶椎',
                    swap='',
                    type=''),
                3:
                dict(color=[
                    34,
                    139,
                    34,
                ], id=3, name='眼', swap='', type=''),
                4:
                dict(color=[
                    139,
                    0,
                    0,
                ], id=4, name='口', swap='', type=''),
                5:
                dict(color=[
                    255,
                    165,
                    0,
                ], id=5, name='肝', swap='', type=''),
                6:
                dict(
                    color=[
                        255,
                        0,
                        255,
                    ], id=6, name='对屏尖', swap='', type=''),
                7:
                dict(color=[
                    255,
                    255,
                    0,
                ], id=7, name='心', swap='', type=''),
                8:
                dict(color=[
                    29,
                    123,
                    243,
                ], id=8, name='肺', swap='', type=''),
                9:
                dict(color=[
                    0,
                    255,
                    255,
                ], id=9, name='肺2', swap='', type='')
            }),
            paper_info=dict(
                author='Tongji Zihao',
                container='OpenMMLab',
                homepage='https://space.bilibili.com/1900783',
                title='Triangle Keypoints Detection',
                year='2023'),
            sigmas=[
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
            ],
            skeleton_info=dict({
                0:
                dict(color=[
                    100,
                    150,
                    200,
                ], id=0, link=(
                    '眼',
                    '扁桃体',
                )),
                1:
                dict(color=[
                    200,
                    100,
                    150,
                ], id=1, link=(
                    '耳门',
                    '听宫',
                )),
                2:
                dict(color=[
                    150,
                    120,
                    100,
                ], id=2, link=(
                    '听宫',
                    '听会',
                )),
                3:
                dict(color=[
                    66,
                    77,
                    229,
                ], id=3, link=(
                    '耳门',
                    '听会',
                ))
            })),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='ear_pose/Ear210_Keypoint_Dataset_coco/val_coco.json',
        type='CocoMetric'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(keypoint_indices=[
        1,
        2,
    ], norm_mode='keypoint_distance', type='NME'),
]
train_batch_size = 32
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='train_coco.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='ear_pose/Ear210_Keypoint_Dataset_coco/',
        metainfo=dict(
            classes='ear',
            dataset_name='Ear210_Keypoint_Dataset_coco',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        101,
                        205,
                        228,
                    ],
                    id=0,
                    name='肾上腺',
                    swap='',
                    type=''),
                1:
                dict(
                    color=[
                        240,
                        128,
                        128,
                    ], id=1, name='耳尖', swap='', type=''),
                10:
                dict(
                    color=[
                        128,
                        0,
                        128,
                    ], id=10, name='膀胱', swap='', type=''),
                11:
                dict(color=[
                    74,
                    181,
                    57,
                ], id=11, name='脾', swap='', type=''),
                12:
                dict(
                    color=[
                        165,
                        42,
                        42,
                    ], id=12, name='角窝中', swap='', type=''),
                13:
                dict(
                    color=[
                        128,
                        128,
                        0,
                    ], id=13, name='神门', swap='', type=''),
                14:
                dict(color=[
                    255,
                    0,
                    0,
                ], id=14, name='肾', swap='', type=''),
                15:
                dict(
                    color=[
                        34,
                        139,
                        34,
                    ], id=15, name='耳门', swap='', type=''),
                16:
                dict(
                    color=[
                        255,
                        129,
                        0,
                    ], id=16, name='听宫', swap='', type=''),
                17:
                dict(
                    color=[
                        70,
                        130,
                        180,
                    ], id=17, name='听会', swap='', type=''),
                18:
                dict(
                    color=[
                        63,
                        103,
                        165,
                    ], id=18, name='肩', swap='', type=''),
                19:
                dict(
                    color=[
                        66,
                        77,
                        229,
                    ], id=19, name='扁桃体', swap='', type=''),
                2:
                dict(color=[
                    154,
                    205,
                    50,
                ], id=2, name='胃', swap='', type=''),
                20:
                dict(
                    color=[
                        255,
                        105,
                        180,
                    ],
                    id=20,
                    name='腰骶椎',
                    swap='',
                    type=''),
                3:
                dict(color=[
                    34,
                    139,
                    34,
                ], id=3, name='眼', swap='', type=''),
                4:
                dict(color=[
                    139,
                    0,
                    0,
                ], id=4, name='口', swap='', type=''),
                5:
                dict(color=[
                    255,
                    165,
                    0,
                ], id=5, name='肝', swap='', type=''),
                6:
                dict(
                    color=[
                        255,
                        0,
                        255,
                    ], id=6, name='对屏尖', swap='', type=''),
                7:
                dict(color=[
                    255,
                    255,
                    0,
                ], id=7, name='心', swap='', type=''),
                8:
                dict(color=[
                    29,
                    123,
                    243,
                ], id=8, name='肺', swap='', type=''),
                9:
                dict(color=[
                    0,
                    255,
                    255,
                ], id=9, name='肺2', swap='', type='')
            }),
            paper_info=dict(
                author='Tongji Zihao',
                container='OpenMMLab',
                homepage='https://space.bilibili.com/1900783',
                title='Triangle Keypoints Detection',
                year='2023'),
            sigmas=[
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
            ],
            skeleton_info=dict({
                0:
                dict(color=[
                    100,
                    150,
                    200,
                ], id=0, link=(
                    '眼',
                    '扁桃体',
                )),
                1:
                dict(color=[
                    200,
                    100,
                    150,
                ], id=1, link=(
                    '耳门',
                    '听宫',
                )),
                2:
                dict(color=[
                    150,
                    120,
                    100,
                ], id=2, link=(
                    '听宫',
                    '听会',
                )),
                3:
                dict(color=[
                    66,
                    77,
                    229,
                ], id=3, link=(
                    '耳门',
                    '听会',
                ))
            })),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=30,
                scale_factor=[
                    0.8,
                    1.2,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.5, type='ChannelShuffle'),
                    dict(p=0.5, type='CLAHE'),
                    dict(p=0.5, type='ColorJitter'),
                    dict(
                        max_height=0.3,
                        max_holes=4,
                        max_width=0.3,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        256,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        12,
                        12,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=30,
        scale_factor=[
            0.8,
            1.2,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.5, type='ChannelShuffle'),
            dict(p=0.5, type='CLAHE'),
            dict(p=0.5, type='ColorJitter'),
            dict(
                max_height=0.3,
                max_holes=4,
                max_width=0.3,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                256,
                256,
            ),
            normalize=False,
            sigma=(
                12,
                12,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                256,
                256,
            ),
            normalize=False,
            sigma=(
                12,
                12,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_batch_size = 8
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='val_coco.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='ear_pose/Ear210_Keypoint_Dataset_coco/',
        metainfo=dict(
            classes='ear',
            dataset_name='Ear210_Keypoint_Dataset_coco',
            joint_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        101,
                        205,
                        228,
                    ],
                    id=0,
                    name='肾上腺',
                    swap='',
                    type=''),
                1:
                dict(
                    color=[
                        240,
                        128,
                        128,
                    ], id=1, name='耳尖', swap='', type=''),
                10:
                dict(
                    color=[
                        128,
                        0,
                        128,
                    ], id=10, name='膀胱', swap='', type=''),
                11:
                dict(color=[
                    74,
                    181,
                    57,
                ], id=11, name='脾', swap='', type=''),
                12:
                dict(
                    color=[
                        165,
                        42,
                        42,
                    ], id=12, name='角窝中', swap='', type=''),
                13:
                dict(
                    color=[
                        128,
                        128,
                        0,
                    ], id=13, name='神门', swap='', type=''),
                14:
                dict(color=[
                    255,
                    0,
                    0,
                ], id=14, name='肾', swap='', type=''),
                15:
                dict(
                    color=[
                        34,
                        139,
                        34,
                    ], id=15, name='耳门', swap='', type=''),
                16:
                dict(
                    color=[
                        255,
                        129,
                        0,
                    ], id=16, name='听宫', swap='', type=''),
                17:
                dict(
                    color=[
                        70,
                        130,
                        180,
                    ], id=17, name='听会', swap='', type=''),
                18:
                dict(
                    color=[
                        63,
                        103,
                        165,
                    ], id=18, name='肩', swap='', type=''),
                19:
                dict(
                    color=[
                        66,
                        77,
                        229,
                    ], id=19, name='扁桃体', swap='', type=''),
                2:
                dict(color=[
                    154,
                    205,
                    50,
                ], id=2, name='胃', swap='', type=''),
                20:
                dict(
                    color=[
                        255,
                        105,
                        180,
                    ],
                    id=20,
                    name='腰骶椎',
                    swap='',
                    type=''),
                3:
                dict(color=[
                    34,
                    139,
                    34,
                ], id=3, name='眼', swap='', type=''),
                4:
                dict(color=[
                    139,
                    0,
                    0,
                ], id=4, name='口', swap='', type=''),
                5:
                dict(color=[
                    255,
                    165,
                    0,
                ], id=5, name='肝', swap='', type=''),
                6:
                dict(
                    color=[
                        255,
                        0,
                        255,
                    ], id=6, name='对屏尖', swap='', type=''),
                7:
                dict(color=[
                    255,
                    255,
                    0,
                ], id=7, name='心', swap='', type=''),
                8:
                dict(color=[
                    29,
                    123,
                    243,
                ], id=8, name='肺', swap='', type=''),
                9:
                dict(color=[
                    0,
                    255,
                    255,
                ], id=9, name='肺2', swap='', type='')
            }),
            paper_info=dict(
                author='Tongji Zihao',
                container='OpenMMLab',
                homepage='https://space.bilibili.com/1900783',
                title='Triangle Keypoints Detection',
                year='2023'),
            sigmas=[
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
            ],
            skeleton_info=dict({
                0:
                dict(color=[
                    100,
                    150,
                    200,
                ], id=0, link=(
                    '眼',
                    '扁桃体',
                )),
                1:
                dict(color=[
                    200,
                    100,
                    150,
                ], id=1, link=(
                    '耳门',
                    '听宫',
                )),
                2:
                dict(color=[
                    150,
                    120,
                    100,
                ], id=2, link=(
                    '听宫',
                    '听会',
                )),
                3:
                dict(color=[
                    66,
                    77,
                    229,
                ], id=3, link=(
                    '耳门',
                    '听会',
                ))
            })),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file='ear_pose/Ear210_Keypoint_Dataset_coco/val_coco.json',
        type='CocoMetric'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(keypoint_indices=[
        1,
        2,
    ], norm_mode='keypoint_distance', type='NME'),
]
val_interval = 10
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(_scope_='mmpose', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpose',
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/rtmpose-s-ear'
