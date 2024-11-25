# Used for training a detector on the nuScenes dataset for the OOD task
# Usually this is intended for the base detector
# Only 10 classes are used. Frames containing classes, which are not part of this set are removed

from mmengine import read_base

from mmood3d.datasets import NuScenesOODDataset
from mmdet3d.datasets.transforms.transforms_3d import (  # noqa
    GlobalRotScaleTrans, ObjectNameFilter, ObjectRangeFilter, PointShuffle,
    PointsRangeFilter, RandomFlip3D)

from mmood3d.evaluation.metrics.nuscenes_metric import CustomNuScenesMetric

with read_base():
    from ..._base_.datasets.nus_3d import *

known_classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

unknown_classes = ['animal', 'debris', 'pushable_pullable', 'personal_mobility', 'stroller', 'wheelchair',
                   'bicycle_rack', 'ambulance_vehicle', 'police_vehicle']

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type=LoadPointsFromMultiSweeps,
        sweeps_num=10,
        backend_args=backend_args),
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type=RandomFlip3D, flip_ratio_bev_horizontal=0.5),
    dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectNameFilter, classes=known_classes),
    dict(type=PointShuffle),
    dict(
        type=Pack3DDetInputs, keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

metainfo = dict(classes=known_classes, unknown_classes=unknown_classes)

train_dataloader.update(
    dict(
        dataset=dict(
            type=NuScenesOODDataset,
            metainfo=metainfo,
            filter_unknowns=True))
)
test_dataloader.update(
    dict(
        dataset=dict(
            type=NuScenesOODDataset,
            metainfo=metainfo,
            filter_unknowns=True))
)
val_dataloader.update(
    dict(
        dataset=dict(
            type=NuScenesOODDataset,
            metainfo=metainfo,
            filter_unknowns=True))
)

val_evaluator = dict(
    type=CustomNuScenesMetric,
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator
