# SPDX-License-Identifier: AGPL-3.0

from mmengine.config import read_base

with read_base():
    from ..ood import *

from mmood3d.datasets.transforms.transforms_3d import OODScaleSample
from mmood3d.datasets.transforms.formating import CustomPack3DDetInputs

work_dir = './Paper_Results/ood_equal_scaling'

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
        # test_mode=True,  # New and worse if true
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
        equal_scaling=True,
        small_scale_range=[0.1, 0.5],
        large_scale_range=[1.5, 3.0]),
    dict(type=PointShuffle),
    dict(
        type=CustomPack3DDetInputs, keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_orig_labels_3d'])
]

train_dataloader.update(
    dict(
        dataset=dict(
            pipeline=train_pipeline)))
