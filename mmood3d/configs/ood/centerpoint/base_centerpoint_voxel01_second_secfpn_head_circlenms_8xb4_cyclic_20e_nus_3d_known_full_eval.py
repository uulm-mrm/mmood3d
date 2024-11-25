# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .base_centerpoint_voxel01_second_secfpn_8xb4_cyclic_20e_nus_3d_known import *

model.update(dict(test_cfg=dict(pts=dict(nms_type='circle'))))

test_dataloader.update(
    dict(
        dataset=dict(filter_unknowns=False)))
val_dataloader.update(
    dict(
        dataset=dict(filter_unknowns=False)))
