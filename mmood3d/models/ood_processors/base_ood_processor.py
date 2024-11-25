# SPDX-License-Identifier: AGPL-3.0

import torch
from mmdet3d.structures.det3d_data_sample import SampleList
from torch import Tensor

from mmood3d.registry import MODELS


@MODELS.register_module()
class BaseOODProcessor:

    def __init(self, threshold: float = 0.0):
        self.threshold = threshold

    def score(self, data: Tensor):
        return 0

    def process(self, detections: SampleList):
        for detection in detections:
            logits_3d = detection.logits_3d
            ood_labels = logits_3d.new_zeros(logits_3d.shape[0], dtype=torch.bool)

            # check if detections already provide ood score, otherwise calculate the ood score
            if 'ood_scores_3d' in detection:
                ood_scores = detection.ood_scores_3d
            else:
                ood_scores = self.score(logits_3d)

            ood_labels[ood_scores >= self.threshold] = 1
            detection.ood_labels_3d = ood_labels

        return detections

