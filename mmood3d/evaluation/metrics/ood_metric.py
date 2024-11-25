# SPDX-License-Identifier: AGPL-3.0

from typing import Dict, List, Sequence

import numpy as np
import torch
import os
from mmengine import print_log
from mmengine.evaluator import BaseMetric
from mmengine.visualization import Visualizer
from mmengine.logging import MessageHub
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from terminaltables import AsciiTable

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

from mmood3d.evaluation.functional.ood_eval import calculate_ood_metrics
from mmood3d.models.ood_processors import BaseOODProcessor
from mmood3d.registry import METRICS, TASK_UTILS, MODELS
from mmood3d.utils import figure_to_image


def safe_stack(data, key):
    if not data:
        return torch.tensor([])
    return torch.stack([item[key] for item in data])

@METRICS.register_module()
class OODMetric(BaseMetric):

    def __init__(self,
                 known_classes: List[str],
                 unknown_classes: List[str],
                 iou_calculator: dict = None,
                 iou_thresholds: List[float] = [],
                 baselines: List[BaseOODProcessor] = [],
                 matching: str = 'iou',
                 assigner: dict = None,
                 sort_predictions: bool = True,
                 plot: bool = True,
                 *args, **kwargs) -> None:
        self.default_prefix = 'OOD metric'
        super(OODMetric, self).__init__(*args, **kwargs)

        if iou_calculator is not None:
            self.iou_calculator = TASK_UTILS.build(iou_calculator)

        self.iou_thresholds = iou_thresholds
        self.known_classes = known_classes
        self.unknown_classes = unknown_classes
        self.sort_predictions = sort_predictions
        self.plot = plot
        
        self.ood_label = len(known_classes)
        
        self.visualizer = Visualizer.get_current_instance()
        self.message_hub = MessageHub.get_current_instance()
        
        self.baselines = [MODELS.build(baseline) for baseline in baselines]
        self.matching = matching

        if assigner is not None:
            self.assigner = TASK_UTILS.build(assigner)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            gt_3d = data_sample['eval_ann_info']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].cpu()
            result['pred_instances_3d'] = pred_3d
            result['gt_instances_3d'] = gt_3d
            result['sample_idx'] = data_sample['sample_idx']

            self.results.append(result)

    def match_boxes_assigner(self, pred_instances, gt_instances):
        matched_boxes = []

        assign_result = self.assigner.assign(pred_instances, gt_instances)
        pos_inds = torch.nonzero(
                assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        for box_ind, gt_ind in zip(pos_inds, pos_assigned_gt_inds):
            matched_boxes.append((gt_ind.item(), box_ind.item()))

        return matched_boxes

    def match_boxes_iou(self, pred_instances, gt_instances):
        matched_boxes = []

        pred_boxes_3d = pred_instances.bboxes_3d
        gt_bboxes_3d = gt_instances.bboxes_3d

        overlaps = self.iou_calculator(pred_boxes_3d, gt_bboxes_3d)
        gt_overlaps = torch.zeros(len(gt_bboxes_3d))
        for j in range(min(len(pred_boxes_3d), len(gt_bboxes_3d))):
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            box_ind = argmax_overlaps[gt_ind]
            gt_overlaps[j] = overlaps[box_ind, gt_ind]

            # cls_iou_thr = self.iou_thresholds[gt_labels_3d[gt_ind.item()]]
            if gt_overlaps[j] >= 0.5:  # cls_iou_thr:
                matched_boxes.append((gt_ind.item(), box_ind.item()))

            # mark the box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        return matched_boxes

    def match_boxes_distance(self, pred_instances, gt_instances):
        matched_boxes = []
        taken = set()
        dist_th = 0.5

        pred_boxes_3d = pred_instances.bboxes_3d
        gt_bboxes_3d = gt_instances.bboxes_3d

        for pred_idx, pred_box in enumerate(pred_boxes_3d):
            min_dist = float('inf')
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_bboxes_3d):
                # Check if this ground truth box has already been matched
                if gt_idx not in taken:
                    this_distance = np.linalg.norm(gt_box[:2] - pred_box[:2])
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx

            # If the closest match is close enough according to threshold, we have a match!
            if min_dist < dist_th:
                matched_boxes.append((match_gt_idx, pred_idx))
                taken.add(match_gt_idx)

        return matched_boxes

    def assign_id_ood(self, results, match_function):
        id_data = []
        ood_data = []

        num_pred = 0
        num_gt = 0
        
        num_ood_gt = 0
        num_ood_matched = 0

        # Iterate through all samples and match detection to ground truths
        for result in results:
            pred_3d = result['pred_instances_3d']
            pred_instance = InstanceData()
            pred_instance.bboxes_3d = pred_3d['bboxes_3d'].tensor
            pred_instance.labels_3d = pred_3d['labels_3d']
            pred_scores_3d = pred_3d['scores_3d']
            pred_logits_3d = pred_3d['logits_3d']

            pred_ood_scores_3d = None
            if 'ood_scores_3d' in pred_3d:
                pred_ood_scores_3d = pred_3d['ood_scores_3d']
            
            obj_feats = None
            if 'obj_feats' in pred_3d:
                obj_feats = pred_3d['obj_feats']
            
            if self.sort_predictions:
                # Sort detections based on confidence score
                sorted_indices = torch.argsort(pred_scores_3d, descending=True)
                pred_instance.bboxes_3d = pred_instance.bboxes_3d[sorted_indices]
                pred_instance.labels_3d = pred_instance.labels_3d[sorted_indices]
                
                pred_scores_3d = pred_scores_3d[sorted_indices]
                pred_logits_3d = pred_logits_3d[sorted_indices]
                
                if pred_ood_scores_3d is not None:
                    pred_ood_scores_3d = pred_ood_scores_3d[sorted_indices]
                    
                if obj_feats is not None:
                    obj_feats = obj_feats[sorted_indices]

            gt_3d = result['gt_instances_3d']
            gt_instance = InstanceData()
            gt_instance.bboxes_3d = gt_3d['gt_bboxes_3d'].tensor
            gt_instance.labels_3d = torch.from_numpy(gt_3d['gt_labels_3d'])

            num_pred += len(pred_instance)
            num_gt += len(gt_instance)

            matched_boxes = match_function(pred_instance, gt_instance)
            
            num_ood_gt += sum(gt_instance.labels_3d >= self.ood_label)

            temp_id = []
            temp_ood = []
            for gt, pred in matched_boxes:
                # mark known classes as ID or OOD
                data = {'logits': pred_logits_3d[pred], 'score': pred_scores_3d[pred]}
                if pred_ood_scores_3d is not None:
                    data.update({'ood_score': pred_ood_scores_3d[pred]})
                    
                if obj_feats is not None:
                    data.update({'obj_feats': obj_feats[pred]})

                gt_label = gt_instance.labels_3d[gt]
                if gt_label < self.ood_label:
                    temp_id.append(data)
                else:
                    temp_ood.append(data)
                    num_ood_matched += 1

            id_data.append(temp_id)
            ood_data.append(temp_ood)

        print(f'num_pred: {num_pred}, num_gt: {num_gt}')

        id_data = [prediction for samples in id_data for prediction in samples]
        ood_data = [prediction for samples in ood_data for prediction in samples]

        return id_data, ood_data, num_ood_gt, num_ood_matched

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        metric_dict = {}

        if self.matching == 'distance':
            match_function = self.match_boxes_distance
        elif self.matching == 'assigner':
            match_function = self.match_boxes_assigner
        else:
            match_function = self.match_boxes_iou

        id_data, ood_data, num_gt, num_ood_matched = self.assign_id_ood(results, match_function)

        id_logits = safe_stack(id_data, 'logits')
        ood_logits = safe_stack(ood_data, 'logits')

        id_scores_default = safe_stack(id_data, 'score')
        ood_scores_default = safe_stack(ood_data, 'score')

        id_feats = safe_stack(id_data, 'obj_feats')
        ood_feats = safe_stack(ood_data, 'obj_feats')
        
        if self.plot:
            self.plot_features(id_feats, ood_feats)
        
        self.fill_metrics_dict(metric_dict, 'Default Score', -id_scores_default, -ood_scores_default, num_gt, num_ood_matched)
        
        # Method that supports direct evaluation
        if 'ood_score' in id_data[0]:
            id_scores = safe_stack(id_data, 'ood_score')
            ood_scores = safe_stack(ood_data, 'ood_score')
            
            self.fill_metrics_dict(metric_dict, 'OOD Score', id_scores, ood_scores, num_gt, num_ood_matched)
            self.fill_metrics_dict(metric_dict, 'Negative OOD Score', -id_scores, -ood_scores, num_gt, num_ood_matched)

        # Evaluate baseline methods
        for baseline in self.baselines:
            baseline_name = type(baseline).__name__

            id_scores = baseline.score(id_logits)
            ood_scores = baseline.score(ood_logits)
            self.fill_metrics_dict(metric_dict, baseline_name, id_scores, ood_scores, num_gt, num_ood_matched)

        # Print metrics as table
        print(f'Name: {self.prefix} ID matched: {len(id_data)}, OOD matched: {len(ood_data)}')
        self.print_summary(metric_dict, logger)

        # Add auxillary data not part of the metrics
        metric_dict[f'{self.default_prefix}/ID Count'] = len(id_data)
        metric_dict[f'{self.default_prefix}/OOD Count'] = len(ood_data)

        return metric_dict

    def plot_features(self, id_feats, ood_feats, max_samples=5000):
        print(f"Creating t-SNE plot using {max_samples} samples.")
        
        id_feats_np = id_feats.cpu().detach().numpy()
        ood_feats_np = ood_feats.cpu().detach().numpy()

        id_sample_size = min(len(id_feats_np), max_samples)
        ood_sample_size = min(len(ood_feats_np), max_samples)
        
        # Randomly sample from ID and OOD features
        id_indices = np.random.choice(len(id_feats_np), id_sample_size, replace=False)
        ood_indices = np.random.choice(len(ood_feats_np), ood_sample_size, replace=False)
        
        id_feats_sampled = id_feats_np[id_indices]
        ood_feats_sampled = ood_feats_np[ood_indices]
        
        all_feats = np.vstack((id_feats_sampled, ood_feats_sampled))
        
        tsne = TSNE(n_components=2, random_state=42, learning_rate="auto", init="pca")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne_result = tsne.fit_transform(all_feats)
        
        tsne_id_feats = tsne_result[:len(id_feats_sampled)]
        tsne_ood_feats = tsne_result[len(id_feats_sampled):]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_id_feats[:, 0], tsne_id_feats[:, 1], label="ID Features", color='green', alpha=0.6, s=60)
        plt.scatter(tsne_ood_feats[:, 0], tsne_ood_feats[:, 1], label="OOD Features", color='red', alpha=0.6, s=60, marker='x')
        plt.legend()
        plt.title("t-SNE of ID and OOD Features")
        
        epoch = self.message_hub.get_info('epoch')
        self.visualizer.add_image(f"eval/val_features", figure_to_image(plt.gcf()), epoch)

    def plot_distribution(self, name, id_scores, ood_scores):
        id_scores = id_scores.cpu().numpy()
        ood_scores = ood_scores.cpu().numpy()

        epoch = self.message_hub.get_info('epoch')

        fontsize = 11  # 18

        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate and plot kernel density estimates for ID and OOD data
        id_kde = gaussian_kde(id_scores)
        ood_kde = gaussian_kde(ood_scores)

        # Define range for x-axis
        x = np.linspace(min(id_scores.min(), ood_scores.min()),
                        max(id_scores.max(), ood_scores.max()), 500)

        # Plot ID and OOD KDEs with filling for visualization
        ax.fill_between(x, id_kde(x), color="#4e79a7", alpha=0.7, label="ID Data")
        ax.fill_between(x, ood_kde(x), color="#f28e2b", alpha=0.7, label="OOD Data")

        # Add axis labels, title, and legend
        ax.xaxis.set_tick_params(labelsize=int(fontsize * 1.2))
        ax.yaxis.set_tick_params(labelsize=int(fontsize * 1.2))
        ax.set_title(name, fontsize=int(fontsize * 1.8))
        ax.set_xlabel('Score', fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

        self.visualizer.add_image(f"eval/dist/{name}", figure_to_image(plt.gcf()), epoch)

    def plot_roc(self, metrics, name):
        fpr = metrics['ROC_FPR']
        tpr = metrics['ROC_TPR']

        epoch = self.message_hub.get_info('epoch')

        fontsize = 11  # 18

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot ROC curve
        ax.plot(fpr, tpr, color="#4e79a7", lw=2, label=f'ROC curve')

        # Plot random classifier line (the "random bar")
        ax.plot([0, 1], [0, 1], color="#f28e2b", lw=2, linestyle='--', label='Random')

        # Add axis labels, title, and legend
        ax.xaxis.set_tick_params(labelsize=int(fontsize * 1.2))
        ax.yaxis.set_tick_params(labelsize=int(fontsize * 1.2))
        ax.set_title(name, fontsize=int(fontsize * 1.8))
        ax.set_xlabel('False Positive Rate', fontsize=fontsize)
        ax.set_ylabel('True Positive Rate', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

        # Add the ROC plot to the visualizer (or display)
        self.visualizer.add_image(f"eval/ROC/{name}", figure_to_image(plt.gcf()), epoch)

    def fill_metrics_dict(self, metric_dict, name, id_scores, ood_scores, num_gt, num_ood_matched):
        metrics, thresh_fpr, thresh_tpr = calculate_ood_metrics(-id_scores, -ood_scores, name,
                                                                output_dir=os.path.join(os.path.expanduser('~'),
                                                                                        "ood_data"))

        # OOD metrics
        for k, v in metrics.items():
            # skip ROC entries
            if not 'ROC_' in k:
                metric_dict[f'{self.default_prefix}/{name}/{k}'] = float(f'{v * 100:.4f}')

        metric_dict[f'{self.default_prefix}/{name}/FPR Threshold'] = float(f'{thresh_fpr:.4f}')
        metric_dict[f'{self.default_prefix}/{name}/TPR Threshold'] = float(f'{thresh_tpr:.4f}')
        
        metric_dict[f'{self.default_prefix}/{name}/Recall'] = float(f'{(num_ood_matched / num_gt) * 100:.4f}')
        
        if self.plot:
            self.plot_distribution(name, -id_scores, -ood_scores)
            self.plot_roc(metrics, name)
    
    def print_summary(self, metric_dict, logger=None):
        if logger == 'silent':
            return

        header = ['Method', 'FPR', 'AUROC', 'AUPR-Success', 'AUPR-Error', 'Recall']
        table_data = [header]

        # extract method names
        methods = {k.split('/')[1] for k in metric_dict}
        for method in sorted(methods):
            row_data = [method]
            for head in header[1:]:
                value = metric_dict[f'{self.default_prefix}/{method}/{head}']
                row_data.append(f'{value:.2f}')
            table_data.append(row_data)

        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=logger)
