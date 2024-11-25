# Modified from https://github.com/deeplearning-wisc/vos/blob/main/metric_utils.py
import os

import numpy as np
import sklearn.metrics as sk

recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def calculate_ood_metrics(_pos, _neg, method_name="default", output_dir=None):
    pos = _pos.cpu().numpy().reshape((-1, 1))
    neg = _neg.cpu().numpy().reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))

    # don't crash if there are no examples
    if len(pos) == 0 or len(neg) == 0:
        print("No examples provided. This should not happen.")
        return {}, 0, 0

    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] = 1

    auroc = sk.roc_auc_score(labels, examples)
    roc_fpr, roc_tpr, _ = sk.roc_curve(labels, examples)
    aupr_success = sk.average_precision_score(labels, examples)
    aupr_error = sk.average_precision_score(1 - labels, 1 - examples)

    fpr, thres_fpr, tpr, thres_tpr = fpr_and_fdr_at_recall(labels, examples, recall_level=recall_level_default)
    metrics = {'FPR': fpr, 'TPR': tpr, 'AUROC': auroc, 'AUPR-Success': aupr_success, 'AUPR-Error': aupr_error,
               'ROC_FPR': roc_fpr, 'ROC_TPR': roc_tpr}

    if output_dir is not None:
        dat = {"labels": labels, "examples": examples, "fpr_threshold": thres_fpr}
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{method_name}.npy"), dat)

    return metrics, thres_fpr, thres_tpr


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default,
                          pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    # tps
    recall_tps = tps / tps[-1]
    recall_fps = fps / fps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall_tps, fps, tps, thresholds = np.r_[recall_tps[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff_fpr = np.argmin(np.abs(recall_tps - recall_level))
    thres_fpr = thresholds[cutoff_fpr]

    cutoff_tpr = np.argmin(np.abs(recall_fps - (1.0 - recall_level)))
    thres_tpr = thresholds[cutoff_tpr]

    fpr = fps[cutoff_fpr] / (np.sum(np.logical_not(y_true)))
    tpr = tps[cutoff_tpr] / (np.sum(y_true))

    return fpr, thres_fpr, tpr, thres_tpr
