# SPDX-License-Identifier: AGPL-3.0

from mmengine.config import read_base

with read_base():
    from .centerpoint.base_centerpoint_voxel01_second_secfpn_8xb4_cyclic_20e_nus_3d_known import *

from torch.optim.sgd import SGD

from mmengine.visualization import TensorboardVisBackend
from mmdet.models.roi_heads.roi_extractors.generic_roi_extractor import GenericRoIExtractor
from mmood3d.datasets.transforms.transforms_3d import OODScaleSample
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmood3d.models.ood_detectors.ood_detector import OODDetector
from mmood3d.models.ood_heads.ood_head import OODHead
from mmood3d.models.task_modules.assigners.min_dist_assigner import MinDistanceAssigner
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from mmdet3d.engine.hooks.visualization_hook import Det3DVisualizationHook
from mmood3d.models.ood_heads.roi_extractors.bev_point_extractor import BEVPooling
from mmood3d.datasets.transforms.formating import CustomPack3DDetInputs

from mmood3d.evaluation.metrics.ood_metric import OODMetric
from mmood3d.models.ood_processors.ood_processors import MSP, MaxLogit, ODIN, Energy

from mmood3d.models.detectors.ood_centerpoint import OODCenterPoint

from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

known_classes = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
unknown_classes = ('animal', 'debris', 'pushable_pullable', 'personal_mobility', 'stroller', 'wheelchair',
                   'bicycle_rack', 'ambulance_vehicle', 'police_vehicle')

metainfo = dict(classes=known_classes, unknown_classes=unknown_classes)

checkpoint = 'checkpoints/base_centerpoint_voxel01_second_secfpn_8xb4_cyclic_20e_nus_3d_known.pth'

detector = model
data_preprocessor = detector.pop('data_preprocessor')
detector_test_cfg = detector.test_cfg  # save version with original thresholds

detector.update(dict(
    type=OODCenterPoint,  # use version that allows better feature extraction
    pts_middle_encoder=dict(return_middle_feats=True),
    # init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint),  # doesn't work right now
    test_cfg=dict(
        pts=dict(
            score_threshold=0.1,
            nms_thr=0.2))
))
del model

# Seeds used: 1473819345 0 2558520944 3991287482 691466502
randomness = dict(seed=1473819345, deterministic=True)
env_cfg.update(dict(cudnn_benchmark=True))

work_dir = './Paper_Results/ood'

lr = 1e-3
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=SGD, lr=lr, momentum=0.9, weight_decay=1e-4),
)

auto_scale_lr = dict(enable=True, base_batch_size=16)

param_scheduler = dict(
    type=PolyLR,
    begin=0,
    end=5,
    eta_min=1e-6,
    by_epoch=True,
    power=3,
    convert_to_iter_based=True)

train_cfg.update(dict(by_epoch=True, max_epochs=5, val_interval=1))
default_hooks.update(
    dict(checkpoint=dict(type=CheckpointHook, interval=1),
         visualization=dict(type=Det3DVisualizationHook)))

model = dict(
    type=OODDetector,
    base_detector=detector,
    base_detector_ckpt=checkpoint,
    data_preprocessor=data_preprocessor,
    ood_head=dict(
        type=OODHead,
        feature_map='neck_feats',
        roi_extractor=dict(
            type=GenericRoIExtractor,
            aggregation='concat',
            roi_layer=dict(type=BEVPooling, pc_range=point_cloud_range, voxel_size=voxel_size, num_points=1),
            out_channels=512,
            featmap_strides=[8]),
        with_box_features=True,
        with_class_scores=True,
        loss_ood=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0
        )
    ),
    train_cfg=detector.train_cfg,
    test_cfg=detector.test_cfg)

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True),
    #    dict(type=ObjectSample, db_sampler=db_sampler),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type=RandomFlip3D,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectNameFilter, classes=known_classes),
    dict(
        type=OODScaleSample,
        unknown_label=len(known_classes),
        small_prob=0.8,
        equal_scaling=False,
        small_scale_range=[0.1, 0.5],
        large_scale_range=[1.5, 3.0]),
    dict(type=PointShuffle),
    dict(
        type=CustomPack3DDetInputs, keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_orig_labels_3d'])
]

test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,  # important
        backend_args=backend_args),
    dict(
        type=MultiScaleFlipAug3D,
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range)
        ]),
    dict(type=Pack3DDetInputs, keys=['points'])
]

train_dataloader.update(
    dict(dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
        ))))

custom_hooks = [
    #    dict(type=ValLossHook, set_model_eval=True, batch_size=2),
]

# no CBGS
train_dataloader.merge(
    dict(
        _delete_=True,
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type=DefaultSampler, shuffle=True),
        dataset=dict(
            type=NuScenesOODDataset,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=dict(classes=known_classes, unknown_classes=unknown_classes),
            filter_unknowns=True,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))

train_dataloader.update(dict(batch_size=2))

val_evaluator = [
    dict(
        type=OODMetric,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        baselines=[dict(type=MSP), dict(type=MaxLogit), dict(type=ODIN), dict(type=Energy)],
        matching='assigner',
        assigner=dict(
            type=MinDistanceAssigner,
            max_dist_thr=0.5,
            coord_inds=[0, 1],
            norm=2),
        prefix='Assigner'),
]

test_evaluator = val_evaluator

val_dataloader.merge(
    dict(
        dataset=dict(
            # test begin
            #            ann_file='nuscenes_infos_train.pkl',
            #            unknown_frames_only=True,
            # test end
            data_root=data_root,
            pipeline=test_pipeline,
            metainfo=dict(classes=known_classes + unknown_classes, unknown_classes=unknown_classes),
            # filter_unknowns=True)))  #filter to test generated outliers in eval
            filter_unknowns=False)))

test_dataloader = val_dataloader

vis_backends = [dict(type=LocalVisBackend),
                dict(type=TensorboardVisBackend)]
visualizer.update(dict(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer'))
