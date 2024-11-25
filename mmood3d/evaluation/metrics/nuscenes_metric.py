# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Dict, List, Optional

import mmengine
from mmdet3d.evaluation import NuScenesMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class CustomNuScenesMetric(NuScenesMetric):

    def _evaluate_single(
            self,
            result_path: str,
            classes: Optional[List[str]] = None,
            result_name: str = 'pred_instances_3d') -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import NuScenesEval
        from mmood3d.evaluation.metrics.nuscenes_utils import CustomNuScenesDetectionEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = CustomNuScenesDetectionEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True)  # False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in classes:
            if name not in metrics['label_aps']:
                continue
            if name not in metrics['label_tp_errors']:
                continue

            for k, v in metrics['label_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail
