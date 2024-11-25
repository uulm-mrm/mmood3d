import copy
import os

from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, \
    filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval


def filter_by_sample_token(ori_eval_boxes, valid_sample_tokens):
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    print(f"GT before filtering: {len(ori_eval_boxes.sample_tokens)}")
    for sample_token in eval_boxes.sample_tokens:
        if sample_token not in valid_sample_tokens:
            eval_boxes.boxes.pop(sample_token)
    print(f"GT after filtering: {len(eval_boxes.sample_tokens)}")
    return eval_boxes


class CustomNuScenesDetectionEval(DetectionEval):
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)

        #self.sample_tokens = self.pred_boxes.sample_tokens

        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        # assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
        #    "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range,
                                            verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range,
                                          verbose=verbose)

        self.sample_tokens = self.pred_boxes.sample_tokens

        # remove frames
        self.gt_boxes = filter_by_sample_token(self.gt_boxes, self.sample_tokens)

        print(f'num_pred: {len(self.pred_boxes)}, num_gt: {len(self.gt_boxes)}')