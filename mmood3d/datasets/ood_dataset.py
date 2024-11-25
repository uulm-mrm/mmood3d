# SPDX-License-Identifier: AGPL-3.0

import copy
import pickle
from abc import ABCMeta
from collections.abc import Mapping
from typing import Callable, List, Optional, Union

from mmdet3d.datasets import Det3DDataset
from mmengine import MMLogger
from mmengine.dataset import force_full_init
from mmengine.logging import print_log

from mmood3d.registry import DATASETS


@DATASETS.register_module()
class BaseOODDataset(Det3DDataset, metaclass=ABCMeta):
    def __init__(self, data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 filter_unknowns: bool = False,
                 unknown_frames_only: bool = False,
                 test_mode: bool = False,
                 **kwargs) -> None:

        self.filter_unknowns = filter_unknowns
        self.unknown_frames_only = unknown_frames_only
        self.valid_indices = []

        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         metainfo=metainfo,
                         modality=modality,
                         pipeline=pipeline,
                         box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt,
                         test_mode=test_mode,
                         **kwargs)

        # show dataset annotation usage
        logger = MMLogger.get_current_instance()
        logger.info(self.__repr__())

    def filter_data(self) -> List[dict]:
        """Filter annotations and measure execution time.

        Returns:
            List[dict]: Filtered results.
        """
        valid_data_infos = []
        self.valid_indices.clear()

        # Discard all frames not containing unknown objects, used for validation
        if self.unknown_frames_only:
            for i, data_info in enumerate(self.data_list):
                contains_unknowns = any(
                    self.idx_to_class[instance['bbox_label_3d']] in self._metainfo['unknown_classes']
                    for instance in data_info['instances']
                )
                if contains_unknowns:
                    valid_data_infos.append(data_info)
                    self.valid_indices.append(i)
        else:
            # Default path
            for i, data_info in enumerate(self.data_list):
                if self.filter_unknowns:
                    contains_unknowns = any(
                        self.idx_to_class[instance['bbox_label_3d']] in self._metainfo['unknown_classes']
                        for instance in data_info['instances']
                    )
                    if contains_unknowns:
                        continue
                valid_data_infos.append(data_info)
                self.valid_indices.append(i)

        return valid_data_infos

    
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = self.valid_indices[idx]
        else:
            data_info['sample_idx'] = len(self) + self.valid_indices[idx]

        return data_info

    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        # not required anymore
        # input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)

        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None

        if hasattr(self, 'show_ins_var') and self.show_ins_var:
            if 'ann_info' in input_dict:
                self._show_ins_var(
                    input_dict['ann_info']['gt_labels_3d'],
                    example['data_samples'].gt_instances_3d.labels_3d)
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    'the data is not in training mode',
                    'current',
                    level=30)

        return example

    @property
    def class_to_idx(self) -> Mapping:
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """
        return {_class: self.label_mapping[i] if _class not in self._metainfo['unknown_classes'] else i
                for i, _class in enumerate(self.METAINFO['classes'])}

    @property
    def idx_to_class(self) -> Mapping:
        """Map class index to mapping class name.

        Returns:
            dict: mapping from class index to class name.
        """
        idx_to_class_mapping = {self.label_mapping[i] if _class not in self._metainfo['unknown_classes'] else i: _class
                for i, _class in enumerate(self.METAINFO['classes'])}
        idx_to_class_mapping[-1] = ''

        return idx_to_class_mapping
