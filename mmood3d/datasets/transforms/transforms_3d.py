# SPDX-License-Identifier: AGPL-3.0

from typing import List

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops


@TRANSFORMS.register_module()
class OODScaleSample(BaseTransform):
    """Randomly scales ID objects and marks them as OOD.

    Required Keys:

    - points
    - gt_bboxes_3d

    Modified Keys:

    - points
    - gt_bboxes_3d
    - gt_labels_3d

    Args:
        unknown_label (int): The label to assign to OOD objects.
            Defaults to 10.
        small_prob (float): Probability of applying smaller scaling to an object.
            Defaults to 0.8.
        equal_scaling (bool): If True, scale uniformly across all dimensions.
            Defaults to False.
        small_scale_range (list[float]): Range of scaling factors when shrinking objects.
            Defaults to [0.1, 0.5].
        large_scale_range (list[float]): Range of scaling factors when enlarging objects.
            Defaults to [1.5, 3.0].
    """

    def __init__(self,
                 unknown_label: int = 10,
                 small_prob: float = 0.8,
                 equal_scaling: bool = False,
                 small_scale_range: List[float] = [0.1, 0.5],
                 large_scale_range: List[float] = [1.5, 3.0]) -> None:
        self.unknown_label = unknown_label
        self.small_prob = small_prob
        self.equal_scaling = equal_scaling
        self.small_scale_range = small_scale_range
        self.large_scale_range = large_scale_range

    def transform(self, input_dict: dict) -> dict:
        """Transform function to apply noise to each ground truth in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each object,
            'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        if isinstance(input_dict, list):
            gt_bboxes_3d = input_dict[0]['eval_ann_info']['gt_bboxes_3d']
            gt_labels_3d = input_dict[0]['eval_ann_info']['gt_labels_3d'].copy()
            points = input_dict[0]['points']
        else:
            gt_bboxes_3d = input_dict['gt_bboxes_3d']
            gt_labels_3d = input_dict['gt_labels_3d'].copy()
            points = input_dict['points']

        # TODO: this is inplace operation
        numpy_box = gt_bboxes_3d.numpy()
        numpy_points = points.numpy()

        independent_scaling = True

        num_rand = 3 if independent_scaling else 1

        origin = (0.5, 0.5, 0)
        gt_box_corners = box_np_ops.center_to_corner_box3d(
            numpy_box[:, :3],
            numpy_box[:, 3:6],
            numpy_box[:, 6],
            origin=origin,
            axis=2)

        surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
        point_masks = box_np_ops.points_in_convex_polygon_3d_jit(numpy_points[:, :3], surfaces)

        selected_boxes = np.zeros(len(numpy_box), dtype=bool)

        # TODO: check collision
        for idx, box in enumerate(numpy_box):
            mask = point_masks[:, idx]
            if np.sum(mask) < 5:
                continue            
            if np.random.rand() < 0.5:
                continue

            center_box = box[:3].copy()
            small_scale_factors = np.random.uniform(self.small_scale_range[0], self.small_scale_range[1], 3)
            large_scale_factors = np.random.uniform(self.large_scale_range[0], self.large_scale_range[1], 3)

            if self.equal_scaling:
                if np.random.rand() < self.small_prob:
                    scale_factors = small_scale_factors
                else:
                    scale_factors = large_scale_factors

                scale_factors = scale_factors[0]
            else:
                random_choice = np.random.rand(num_rand) > self.small_prob
                scale_factors = np.where(random_choice, large_scale_factors, small_scale_factors)

            numpy_points[mask, :3] -= center_box
            numpy_points[mask, :3] *= scale_factors
            numpy_points[mask, :3] += center_box[:3]

            # numpy_box[idx, 0:3] -= center_box
            numpy_box[idx, 3:6] *= scale_factors
            # numpy_box[idx, 0:3] += center_box
            
            selected_boxes[idx] = 1

        if isinstance(input_dict, list):
            input_dict[0]['eval_ann_info']['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
            input_dict[0]['eval_ann_info']['gt_orig_labels_3d'] = gt_labels_3d
            input_dict[0]['eval_ann_info']['gt_labels_3d'][selected_boxes] = self.unknown_label
            input_dict[0]['points'] = points.new_point(numpy_points)

        else:
            input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
            input_dict['gt_orig_labels_3d'] = gt_labels_3d
            input_dict['gt_labels_3d'][selected_boxes] = self.unknown_label
            input_dict['points'] = points.new_point(numpy_points)

        return input_dict
