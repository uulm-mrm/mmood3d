# SPDX-License-Identifier: AGPL-3.0

from mmood3d.registry import DATASETS

from .ood_dataset import BaseOODDataset
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesOODDataset(BaseOODDataset, NuScenesDataset):
    METAINFO = {
        'classes': ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier', 'animal', 'debris', 'pushable_pullable',
                    'personal_mobility', 'stroller', 'wheelchair', 'bicycle_rack', 'ambulance_vehicle',
                    'police_vehicle'),
        'known_classes': ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                          'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'unknown_classes': ('animal', 'debris', 'pushable_pullable', 'personal_mobility', 'stroller', 'wheelchair',
                            'bicycle_rack', 'ambulance_vehicle', 'police_vehicle'),
        'version': 'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey

            (70, 130, 180),  # Steelblue
            (210, 105, 30),  # Chocolate
            (105, 105, 105),  # Dimgrey
            (219, 112, 147),  # Palevioletred
            (240, 128, 128),  # Lightcoral
            (138, 43, 226),  # Blueviolet
            (188, 143, 143),  # Rosybrown
            (255, 83, 0),
            (255, 215, 0),  # Gold
        ]
    }

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs)
