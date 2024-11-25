# SPDX-License-Identifier: AGPL-3.0

import torch
from torch import Tensor
from .base_ood_processor import BaseOODProcessor
import torch.nn.functional as F

from mmood3d.registry import MODELS


@MODELS.register_module()
class MSP(BaseOODProcessor):
    """
    Maximum Softmax Probability (MSP) score.

    Reference:
        - Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting 
          Misclassified and Out-of-Distribution Examples in Neural Networks." 
          In International Conference on Learning Representations (ICLR).
    """

    def score(self, data: Tensor):
        return -F.softmax(data, dim=1).max(1)[0]


@MODELS.register_module()
class MaxLogit(BaseOODProcessor):
    """
    Maximum Logit score.

    Reference:
        - Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting 
          Misclassified and Out-of-Distribution Examples in Neural Networks." 
          In International Conference on Learning Representations (ICLR).
    """

    def score(self, data: Tensor):
        return -data.max(1)[0]


@MODELS.register_module()
class ODIN(BaseOODProcessor):
    """
    ODIN score.

    Reference:
        - Liang, S., Li, Y., & Srikant, R. (2018). "Enhancing The Reliability 
          of Out-of-distribution Image Detection in Neural Networks." In 
          International Conference on Learning Representations (ICLR).
    """

    def __init__(self, temperature: float = 1000):
        self.temperature = temperature

    def score(self, data: Tensor):
        return -F.softmax(data / self.temperature, dim=1).max(1)[0]


@MODELS.register_module()
class Energy(BaseOODProcessor):
    """
    Energy-based score.

    Reference:
        - Liu, W., Wang, X., Owens, J. D., & Li, Y. (2020). "Energy-based 
          Out-of-distribution Detection." In Advances in Neural Information 
          Processing Systems (NeurIPS).
    """

    def __init__(self, temperature: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def score(self, data: Tensor):
        return -self.temperature * torch.logsumexp(data / self.temperature, dim=-1)

