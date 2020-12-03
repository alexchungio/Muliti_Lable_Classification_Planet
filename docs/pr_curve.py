#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : pr_curve.py
# @ Description: https://cloud.tencent.com/developer/article/1559255
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/3 上午10:26
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def get_precision_and_recall(y_label, y_prob):
    """

    :param y_label: true label
    :param y_score: predict probability
    :param n:
    :return:
    """
    assert len(y_label) == len(y_prob)
    # invert sort y_pred
    score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_prob = np.array(y_prob)[score_indices]
    y_true = np.array(y_label)[score_indices]

    # ------------------get tps and fps at distinct value -------------------------
    # extract the indices associated with the distinct values
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]

    # computer false positive
    fps = threshold_idxs + 1 - tps

    # get distinct value prob
    thresholds = y_prob[threshold_idxs]

    #------------------------ computer precision and recall------------------------
    # computer precision
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    # computer recall
    recall = tps / tps[-1]

    # stop when full recall attained
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind+1)

    #------------------------ computer AP------------------------------------------
    ap  = 0.
    for i, (l_r, h_r) in enumerate(zip(recall[:-1], recall[1:])):

        ap += 1 / 2 * (h_r - l_r) * (precision[i] + precision[i+1])

    ap = round(ap, 4)

    return np.r_[1, precision[sl]], np.r_[0, recall[sl]], ap


def pr_plot(precision, recall, area):

    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, linestyle='-', linewidth=2,
             label='Precision-Recall Curve Area={}'.format(area))
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=15)
    plt.legend(loc='best')


def main():
    y_label = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]

    y_prob = [-0.20079125, 0.30423529, 0.2010557, 0.27523383, 0.42592946, -0.15043958,
              -0.08794977, -0.12733765, 0.22931154, -0.23913774, -0.0638661, -0.14958713,
              -0.04915145, 0.09898199, 0.05155884, -0.1142967, 0.16105883, 0.04871601,
              -0.08258422, -0.26105925]

    precision, recall, ap = get_precision_and_recall(y_label, y_prob)
    pr_plot(precision, recall, ap)

    plt.show()
    print('Done')

if __name__ == "__main__":
    main()

