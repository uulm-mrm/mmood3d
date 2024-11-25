# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet3d.models import CenterHead
from mmdet3d.models.layers import circle_nms, nms_bev
from mmdet3d.structures import xywhr2xyxyr
from mmengine.logging import MessageHub
from mmengine.structures import InstanceData

from mmood3d.registry import MODELS


@MODELS.register_module()
class CustomCenterHead(CenterHead):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.message_hub = MessageHub.get_current_instance()

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]],
                        batch_input_metas: List[dict], *args,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_input_metas (list[dict]): Meta info of multiple
                inputs.

        Returns:
            list[:obj:`InstanceData`]: Instance prediction
            results of each sample after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (:obj:`LiDARInstance3DBoxes`): Prediction
                  of bboxes, contains a tensor with shape
                  (num_instances, 7) or (num_instances, 9), and
                  the last 2 dimensions of 9 is
                  velocity.
        """

        # Combine task specific heatmaps into one
        batch_heatmap_raw = torch.cat([preds_dict[0]['heatmap'] for preds_dict in preds_dicts], dim=1)
        self.message_hub.update_info('centerpoint_head:batch_heatmap_raw', batch_heatmap_raw)

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_heatmap_raw,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            batch_cls_logits = [box['logits'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    logits = temp[i]['logits']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    logits = logits[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels, logits=logits)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels,
                                             batch_cls_logits,
                                             batch_input_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'logits':
                    logits = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            temp_instances.logits_3d = logits
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, batch_cls_logits, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            batch_cls_logits (list[torch.Tensor]): Prediction logits with the
                shape of [N, C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
                -logits (torch.Tensor): Prediction logits after nms with the
                    shape of [N, C].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels, logits_preds) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels, batch_cls_logits)):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)
                top_logits = logits_preds.squeeze(-1)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)
                top_logits = logits_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                    top_logits = top_logits[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]
            selected_logits = top_logits[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                logits_preds = selected_logits
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_logits = logits_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask],
                        logits=final_logits[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels,
                        logits=final_logits)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device),
                    logits=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device)
                )

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
