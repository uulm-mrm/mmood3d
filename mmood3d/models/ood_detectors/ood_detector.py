# SPDX-License-Identifier: AGPL-3.0

from typing import Union, List, Optional

import torch
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.runner import load_checkpoint
from torch import Tensor, nn

from mmood3d.registry import MODELS


@MODELS.register_module()
class OODDetector(Base3DDetector):
    """Base class of an OOD 3D detector.

    It inherits original ``:class:Base3DDetector``.
    """

    def __init__(
        self,
        base_detector: ConfigType,
        base_detector_ckpt: Optional[str] = None,
        ood_head: OptConfigType = None,
        ood_postprocessor: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super(OODDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.base_detector_cfg = base_detector
        self.base_detector = MODELS.build(base_detector)
        
        self.base_detector_ckpt = base_detector_ckpt
        self.init_base_detector()

        if ood_head is not None:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            ood_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            ood_head.update(test_cfg=pts_test_cfg)
            self.ood_head = MODELS.build(ood_head)

        if ood_postprocessor is not None:
            self.ood_postprocessor = MODELS.build(ood_postprocessor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_base_detector(self):
        if self.base_detector_ckpt:
            _ = load_checkpoint(self.base_detector, self.base_detector_ckpt)
            # avoid loaded parameters be overwritten
            self.base_detector._is_init = True

            # fixed object detector
            self.freeze_module(self.base_detector)
            
    @property
    def with_ood_head(self):
        """bool: Whether the detector has an OOD head."""
        return hasattr(self, 'ood_head') and self.ood_head is not None

    @property
    def with_ood_postprocessor(self):
        """bool: Whether the detector has an OOD postprocessor."""
        return hasattr(self, 'ood_postprocessor') and self.ood_postprocessor is not None

    def freeze_module(self, module: Union[List[nn.Module], nn.Module]) -> None:
        """Freeze module during training."""
        if isinstance(module, nn.Module):
            modules = [module]
        else:
            modules = module
        for module in modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Overload in order to keep base_detector in eval mode."""
        super().train(mode)
        self.base_detector.eval()

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since base_detector is registered as a plain object, it is necessary
        to put the base_detector to cuda when calling ``cuda`` function."""
        self.base_detector.cuda(device=device)
        return super().cuda(device=device)

    def cpu(self) -> nn.Module:
        """Since base_detector is registered as a plain object, it is necessary
        to put the base_detector to cpu when calling ``cpu`` function."""
        self.base_detector.cpu()
        return super().cpu()

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since base_detector is registered as a plain object, it is necessary
        to put the base_detector to other device when calling ``to``
        function."""
        self.base_detector.to(device=device)
        return super().to(device=device)

#     def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the base_detector model from being registered as a
        nn.Module. The base_detector module is registered as a plain object, so that
        the base_detector parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
 #       if name == 'base_detector':
 #           object.__setattr__(self, name, value)
 #       else:
 #           super().__setattr__(name, value)
    
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """

        losses = dict()

        if self.with_ood_head:
            with torch.no_grad():
                self.base_detector.eval()
                results_list = self.base_detector.predict(batch_inputs_dict, batch_data_samples, **kwargs)
                torch.cuda.empty_cache()
            inter_features = self.base_detector.inter_features

            ood_loss = self.ood_head.loss(results_list, batch_inputs_dict, batch_data_samples, inter_features)
            del inter_features

            losses.update(ood_loss)

        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        with torch.no_grad():
            results_list = self.base_detector.predict(batch_inputs_dict, batch_data_samples, **kwargs)

        inter_features = self.base_detector.inter_features
        pred_instances_3d = [results.pred_instances_3d for results in results_list]

        if self.with_ood_head:
            pred_instances_3d = self.ood_head.predict(results_list, pred_instances_3d, inter_features,
                                                      batch_data_samples)
        
        del inter_features

        predictions = self.add_pred_to_datasample(batch_data_samples, pred_instances_3d)

        if self.with_ood_postprocessor:
            predictions = self.ood_postprocessor.process(predictions)

        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 **kwargs) -> tuple:
        raise NotImplementedError(
            "_forward function (namely 'tensor' mode) is not supported now")