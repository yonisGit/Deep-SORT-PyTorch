# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
TRACKERS = MODELS
MOTION = MODELS
REID = MODELS
AGGREGATORS = MODELS


def build_tracker(cfg):
    """Build tracker."""
    return TRACKERS.build(cfg)


def build_motion(cfg):
    """Build motion model."""
    return MOTION.build(cfg)


def build_reid2(cfg):
    """Build reid model."""
    return REID.build(cfg)


def build_reid():
    """Build reid model."""
    return REID.build(dict(
        type='BaseReID',
        backbone=dict(
            # type='SwinTransformer',
            # pad_small_map=True,
            # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=None, stride=None),
        # neck=dict(in_channels=[96, 192, 384, 768]),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            # in_channels=768,
            # fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            # loss_pairwise=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
            # 'pths/resnet50_msmt17.pth'
            # 'pths/resnet5011.pth'
            # 'pths/swin.pth'
        )))


def build_aggregator(cfg):
    """Build aggregator model."""
    return AGGREGATORS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is None and test_cfg is None:
        return MODELS.build(cfg)
    else:
        return MODELS.build(cfg, MODELS,
                            dict(train_cfg=train_cfg, test_cfg=test_cfg))
