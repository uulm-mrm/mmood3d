# SPDX-License-Identifier: AGPL-3.0

from mmengine.config import read_base

with read_base():
    from ..ood import *

work_dir = './Paper_Results/ood_safe'

model.update(dict(
    ood_head=dict(
        feature_map='bev_multi_scale_feats',
        roi_extractor=dict(
            out_channels=32 + 64 + 128 + 128,
            featmap_strides=[2, 4, 8, 8]))))
