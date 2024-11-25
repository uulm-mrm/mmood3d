# SPDX-License-Identifier: AGPL-3.0

from mmengine.config import read_base

with read_base():
    from ..ood import *

work_dir = './Paper_Results/ablation_cls_feats'

model.update(dict(
    ood_head=dict(
        with_box_features=False,
        with_class_scores=True)))