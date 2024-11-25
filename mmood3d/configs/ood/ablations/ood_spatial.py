# SPDX-License-Identifier: AGPL-3.0

from mmengine.config import read_base

with read_base():
    from ..ood import *

work_dir = './Paper_Results/ood_spatial'

model.update(dict(
    ood_head=dict(
        feature_map='spatial_feats',
        roi_extractor=dict(
            out_channels=256,
            featmap_strides=[8]))))
