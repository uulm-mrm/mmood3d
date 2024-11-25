from typing import List

import numpy as np
import torch
from mmdet3d.structures import rotation_3d_in_axis
from mmengine.model import BaseModule
from torch.nn import functional as F

from mmood3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BEVPooling(BaseModule):
    def __init__(
            self,
            pc_range: List[float],
            voxel_size: List[float],
            spatial_scale: float,
            num_points: int = 5,
    ):
        super(BEVPooling, self).__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.spatial_scale = spatial_scale
        self.num_points = num_points
        
        self.output_size = (num_points,)

    def forward(self, features: torch.Tensor, rois: torch.Tensor):
        roi_features = []
        batch_size = features.shape[0]

        for batch_idx in range(batch_size):
            batch_mask = rois[:, 0] == batch_idx
            batch_rois = rois[batch_mask, 1:]  # remove batch dim
            num_boxes = len(batch_rois)

            feature_points = self.get_feature_points(batch_rois)

            xs = (feature_points[..., 0] - self.pc_range[0]) / self.voxel_size[0]
            xs *= self.spatial_scale
            ys = (feature_points[..., 1] - self.pc_range[1]) / self.voxel_size[1]
            ys *= self.spatial_scale

            feats = self.bilinear_interpolate(features[batch_idx], xs, ys)
            
            roi_feature = torch.cat([feats[i * num_boxes:(i + 1) * num_boxes] for i in range(self.num_points)], dim=1)
            
#            roi_feature = torch.stack([feats[i * num_boxes:(i + 1) * num_boxes] for i in range(self.num_points)], dim=1)
            
            roi_features.append(roi_feature)

        roi_features = torch.cat(roi_features, dim=0).unsqueeze(-1)
#        roi_features = torch.cat(roi_features, dim=0)
        return roi_features

    def get_feature_points(self, boxes: torch.Tensor):
        if self.num_points == 5:
            corners = self.center_to_corner_box2d(boxes[:, :2], boxes[:, 3:5], boxes[:, 6])

            center = boxes[:, :2]
            front = (corners[:, 0] + corners[:, 1]) / 2.0
            back = (corners[:, 2] + corners[:, 3]) / 2.0
            left = (corners[:, 0] + corners[:, 3]) / 2.0
            right = (corners[:, 1] + corners[:, 2]) / 2.0
            points = torch.cat([center, front, back, left, right], dim=0)
            return points
        elif self.num_points == 1:
            center = boxes[:, :2]
            return center
        else:
            raise NotImplementedError()

    @staticmethod
    def center_to_corner_box2d(centers: torch.Tensor,
                               dims: torch.Tensor,
                               angles: torch.Tensor,
                               origin = 0.5) -> torch.Tensor:
        ndim = dims.shape[1]
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1)).to(
            device=dims.device, dtype=dims.dtype)
        if ndim == 2:
            # generate clockwise box corners
            corners_norm = corners_norm[[0, 1, 3, 2]]
        elif ndim == 3:
            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - dims.new_tensor(origin)
        corners = dims.view([-1, 1, ndim]) * corners_norm.view([1, 2 ** ndim, ndim])
        corners = rotation_3d_in_axis(corners, angles)
        corners += centers.view([-1, 1, 2])

        return corners

    @staticmethod
    def bilinear_interpolate(features: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        C, H, W = features.shape[-3:]
        x_norm = x / (W - 1) * 2 - 1
        y_norm = y / (H - 1) * 2 - 1
        grid = torch.cat([x_norm.view(-1, 1), y_norm.view(-1, 1)], dim=1)
        grid = grid.unsqueeze(0).unsqueeze(0)

        features = features.unsqueeze(0)

        feat = F.grid_sample(features, grid, mode='bilinear', align_corners=True)
#        feat = F.grid_sample(features, grid, mode='nearest', align_corners=True) # not good performance

        feat = feat.view(C, -1).permute(1, 0)
        return feat
