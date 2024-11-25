from typing import Dict

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.structures import bbox3d2roi
from mmengine import MMLogger
from mmengine.logging import MessageHub
from mmengine.model import BaseModule
from mmengine.visualization import Visualizer
from torch import Tensor
from torch.nn import functional as F

from mmood3d.models.ood_heads.roi_extractors.bev_point_extractor import BEVPooling
from mmood3d.registry import MODELS, TASK_UTILS
from mmood3d.utils import figure_to_image
from mmood3d.utils.plot_utils import plot_features_ood

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class OODHead(BaseModule):

    def __init__(self,
                 head: dict = None,
                 roi_extractor: OptMultiConfig = None,
                 loss_ood: dict = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 feature_map: str = 'neck_feats',
                 fm_select = None,
                 box_englargement: float = 0.0,
                 with_box_features: bool = False,
                 with_class_scores: bool = False,
                 dropout_ratio: float = 0.3,
                 encode_dim: int = 64,
                 ood_label: int = 10,
                 plot_interval: int = 200,
                 normalize_feats: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs
                 ) -> None:
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.roi_extractor = MODELS.build(roi_extractor)
        self.feature_map = feature_map
        self.fm_select = fm_select
        self.dropout_ratio = dropout_ratio
        self.box_englargement = box_englargement
        self.with_box_features = with_box_features
        self.with_class_scores = with_class_scores
        self.ood_label = ood_label
        self.normalize_feats = normalize_feats

        self.plot_interval = plot_interval
        self.logger = MMLogger.get_current_instance()

        self.head = head
        self.head_in = roi_extractor.out_channels
        if self.head is None:
            
            if self.with_box_features:
                self.head_in += encode_dim
            if self.with_class_scores:
                self.head_in += encode_dim
            
            self.head = nn.Sequential(
                nn.Linear(self.head_in, self.head_in // 2),
                #nn.ReLU(),
                nn.Linear(self.head_in // 2, self.head_in // 4),
                #nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(self.head_in // 4, 1),
            )
        else:
            self.head = MODELS.build(head)
        
        self.loss_ood = MODELS.build(loss_ood)

        if self.with_box_features:
            self.box_encoder = nn.Linear(7, encode_dim)

        if self.with_class_scores:
            # ood_label is essentially the number of id classes.
            self.logits_encoder = nn.Linear(ood_label * 2, encode_dim)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.message_hub = MessageHub.get_current_instance()
        self.visualizer = Visualizer.get_current_instance()
        
        self.validating = False

    def to_bev(self, feature_map):
        # from https://github.com/open-mmlab/mmdetection3d/blob/d024a1f9f50cb23e790337406651d5f45691fa39/mmdet3d/models/middle_encoders/sparse_encoder_voxelnext.py
        features_cat = feature_map.features
        indices_cat = feature_map.indices[:, [0, 2, 3]]
        spatial_shape = feature_map.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        bev_map = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=feature_map.batch_size
        )
        return bev_map
    
    def process_inter_features(self, inter_features):
        # Convert sparse 3D features to dense BEV features
        bev_multi_scale_feats = []
        if 'bev_multi_scale_feats' in self.feature_map:
            for feat in inter_features['multi_scale_3d_feats']:
                spatial_features = self.to_bev(feat).dense()
                bev_multi_scale_feats.append(spatial_features)
            inter_features['bev_multi_scale_feats'] = bev_multi_scale_feats

        return inter_features

    def process_predictions(self, results_list):
        pred_instances_3d = [results.pred_instances_3d for results in results_list]
        pred_boxes = [instance.bboxes_3d for instance in pred_instances_3d]
        pred_labels = [instance.labels_3d for instance in pred_instances_3d]
        pred_logits = [instance.logits_3d for instance in pred_instances_3d]

        return pred_boxes, pred_labels, pred_logits
    
    def forward(self, results_list, inter_features):
        inter_features = self.process_inter_features(inter_features)
        pred_boxes, pred_labels, pred_logits = self.process_predictions(results_list)

        if self.box_englargement > 0.0:
            enl_pred_boxes = pred_boxes.enlarged_box(extra_width=self.box_englargement)
        else:
            enl_pred_boxes = pred_boxes
        
        if self.training or self.validating:
            # use gt boxes
            enl_pred_boxes = [res.gt_instances_3d.bboxes_3d.clone() for res in results_list]
                
        rois = bbox3d2roi([box.tensor for box in enl_pred_boxes])

        features = inter_features[self.feature_map]
        features = features if isinstance(features, (list, tuple)) else [features]

        obj_feats = self.roi_extractor(features, rois)
        obj_feats = torch.stack([feat.view(-1) for feat in obj_feats])
        obj_feats = F.normalize(obj_feats, dim=-1) if self.normalize_feats else obj_feats
                    
        cfg = self.train_cfg if self.training else self.test_cfg
        pc_range = cfg.get('point_cloud_range', None)
        pc_range = pc_range if pc_range is not None else cfg.get('pc_range', None)
        out_size_factor = cfg['out_size_factor']
        
        if self.with_class_scores and (self.training or self.validating):
            batch_heatmap_raw = self.message_hub.get_info('centerpoint_head:batch_heatmap_raw')
            logits_pooling = BEVPooling(pc_range=pc_range,
                                        voxel_size=self.train_cfg['voxel_size'],
                                        spatial_scale=1.0 / out_size_factor,
                                        num_points=1)
            gt_logits = logits_pooling(batch_heatmap_raw, rois)
                
        if self.with_box_features:
            if self.training or self.validating:
                pred_boxes = [res.gt_instances_3d.bboxes_3d for res in results_list]
            
            box_feats = torch.cat([box.tensor for box in pred_boxes])
            # remove velocity features (both gt and prediction)
            if box_feats.shape[-1] > 7:
                box_feats = box_feats[:, :7]
            
            add_noise = False
            if add_noise:
                std_dev = 0.1
                noise = torch.randn_like(box_feats) * std_dev
                box_feats += noise
            
            box_feats = self.box_encoder(box_feats)
            box_feats = F.normalize(box_feats, dim=-1) if self.normalize_feats else box_feats
            obj_feats = torch.cat([obj_feats, box_feats], dim=1)
      
        if self.with_class_scores:
            if self.training or self.validating:
                logits = torch.stack([logit.view(-1) for logit in gt_logits])
                # use original labels not containing OOD labels
                pred_labels = [res.gt_instances_3d.orig_labels_3d for res in results_list]
            else:
                logits = torch.cat(pred_logits)
        
            cat_labels = torch.cat(pred_labels)
            one_hot_labels = F.one_hot(cat_labels.long(), num_classes=logits.shape[1])
            
            combined_logits = torch.cat([logits, one_hot_labels], dim=1)
            logits_feats = self.logits_encoder(combined_logits)
            logits_feats = F.normalize(logits_feats, dim=-1) if self.normalize_feats else logits_feats
            obj_feats = torch.cat([obj_feats, logits_feats], dim=1)

        if self.normalize_feats:
            obj_feats = F.normalize(obj_feats, dim=-1)
        
        return self.head(obj_feats), obj_feats

    def predict(self,
                results_list: InstanceList,
                pred_instances_3d_list: InstanceList,
                inter_features: dict,
                batch_data_samples: SampleList) -> InstanceList:

        batch_size = len(pred_instances_3d_list)
        assert batch_size == 1
        
        pred_instance = pred_instances_3d_list[0]

        if len(pred_instance.bboxes_3d) == 0:
            return pred_instances_3d_list

        ood_scores, obj_feats = self.forward(results_list, inter_features)
        ood_scores = ood_scores.squeeze()
        obj_feats = obj_feats.squeeze()

        if ood_scores.dim() == 0:
            ood_scores = ood_scores.unsqueeze(0)

        pred_instance.ood_scores_3d = torch.sigmoid(ood_scores)
        pred_instance.obj_feats = obj_feats

        return pred_instances_3d_list

    def loss(self, results_list: InstanceList,
             batch_inputs_dict: dict,
             batch_data_samples: SampleList,
             inter_features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation

        Args:
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d` .

        Returns:
            dict: Losses of each branch.
        """

        ood_scores, obj_feats = self.forward(results_list, inter_features)

        batch_gt_instances_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)

        labels_3d = torch.cat([inst.labels_3d for inst in batch_gt_instances_3d])
 
        ood_labels = (labels_3d >= self.ood_label).long()
        label_weights = labels_3d.new_ones(labels_3d.shape[0], dtype=torch.float)
        
        loss_ood = self.loss_ood(ood_scores, ood_labels, label_weights)
        
        iteration = self.message_hub.get_info('iter')
        if self.plot_interval > 0 and (iteration % self.plot_interval) == 0:
            ood_plot = plot_features_ood(obj_feats, ood_labels)
            self.visualizer.add_image(f"id_ood_feats", figure_to_image(ood_plot), iteration)

        return dict(loss_ood=loss_ood)
    