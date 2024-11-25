# SPDX-License-Identifier: AGPL-3.0

from typing import Optional, Union, List

import torch
from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmengine.structures import InstanceData
from torch import Tensor

from mmood3d.registry import TASK_UTILS
from mmdet3d.structures import LiDARInstance3DBoxes


def _to_tensor(boxes: Union[Tensor, LiDARInstance3DBoxes]) -> Tensor:
    if isinstance(boxes, LiDARInstance3DBoxes):
        return boxes.tensor
    return boxes


@TASK_UTILS.register_module()
class MinDistanceAssigner(BaseAssigner):
    def __init__(
        self,
        max_dist_thr: float = 0.5,
        coord_inds: List[int] = [0, 1],  # [0, 1, 2],
        norm: int = 1,
        method: int = 0,
    ) -> None:
        self.max_dist_thr = max_dist_thr
        self.coord_inds = coord_inds
        self.norm = norm
        self.method = method

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:

        assert isinstance(gt_instances.labels_3d, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)

        gt_labels = gt_instances.labels_3d
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds,),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds,),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        assigned_vals = torch.full((num_preds,),
                                     10000,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        pred_bboxes = _to_tensor(pred_instances.bboxes_3d)
        gt_bboxes = _to_tensor(gt_instances.bboxes_3d)
        
        if self.method == 0:
            distance = torch.norm(pred_bboxes[:, self.coord_inds][None, :, :] - gt_bboxes[:, self.coord_inds][:, None, :],
                                  dim=-1)  # [num_gts, num_bboxes]
            min_distances, min_distance_inds = distance.min(dim=1)

            for idx_gts in range(num_gts):
                # for idx_pred in torch.where(distance[idx_gts] < self.max_dist_thr)[0]:
                # # each gt match to all the pred box within some radius
                idx_pred = min_distance_inds[idx_gts]  # each gt only match to the nearest pred box
                #if distance[idx_gts, idx_pred] <= self.max_dist_thr:
                if distance[idx_gts, idx_pred] < self.max_dist_thr:
                    # if this pred box is assigned, then compare
                    if distance[idx_gts, idx_pred] < assigned_vals[idx_pred]:
                        assigned_vals[idx_pred] = distance[idx_gts, idx_pred]
                        # for AssignResult, 0 is negative, -1 is ignore, 1-based
                        # indices are positive
                        assigned_gt_inds[idx_pred] = idx_gts + 1
                        assigned_labels[idx_pred] = gt_labels[idx_gts]
        elif self.method == 1:
            distance = torch.cdist(pred_bboxes[:, self.coord_inds], gt_bboxes[:, self.coord_inds], p=self.norm)
            min_distances, min_distance_inds = distance.min(dim=1)

            pos_inds = torch.where(min_distances <= self.max_dist_thr)[0]

            assigned_gt_inds[:] = 0
            assigned_gt_inds[pos_inds] = min_distance_inds[pos_inds]
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        elif self.method == 2:
            distance = torch.cdist(pred_bboxes[:, self.coord_inds], gt_bboxes[:, self.coord_inds], p=self.norm)
            # distance = torch.norm(pred_bboxes[:, None, 0:2] - gt_bboxes[None, :, 0:2], dim=-1)

            within_thr = distance <= self.max_dist_thr
            distance[~within_thr] = float('inf')
            min_distances, min_distance_inds = distance.min(dim=1)

            valid_inds = min_distances != float('inf')
            nearest_gt_inds = min_distance_inds[valid_inds]
            assigned_gt_inds[valid_inds] = nearest_gt_inds + 1
            assigned_labels[valid_inds] = gt_labels[nearest_gt_inds]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=assigned_vals,
            labels=assigned_labels)

