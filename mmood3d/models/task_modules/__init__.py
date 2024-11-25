# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.task_modules import AssignResult, BaseAssigner

from .coders import (CustomCenterPointBBoxCoder,)

__all__ = [
    'BaseAssigner', 'AssignResult', 'CustomCenterPointBBoxCoder',
]
