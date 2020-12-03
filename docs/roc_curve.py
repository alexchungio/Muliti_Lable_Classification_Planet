#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : roc_curve.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/3 下午5:17
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def get_roc(y_label, y_score):


    assert len(y_label) == len(y_score)
    # invert sort y_pred
    score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_prob = np.array(y_score)[score_indices]
    y_true = np.array(y_label)[score_indices]

    # ------------------get tps and fps at distinct value -------------------------
    # extract the indices associated with the distinct values
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]

    # computer false positive
    fps = threshold_idxs + 1 - tps

    # # get distinct value prob
    # thresholds = y_prob[threshold_idxs]
    # thresholds = np.r_[thresholds[0] + 1, thresholds]
    # ------------------------------ computer tpr and fpr---------------------------
    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    # -------------------------------computer auc------------------------------------
    auc = 0.
    for i, (l_r, h_r) in enumerate(zip(fpr[:-1], fpr[1:])):
        auc += 1 / 2 * (h_r - l_r) * (tpr[i] + tpr[i + 1])

    return tpr, fpr, auc


def roc_plot(tpr, fpr, auc):
    plt.figure()

    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = {:.4f})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, color='C0', alpha=0.4, interpolate=True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def main():
    y_label = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]

    y_score = [-0.20079125, 0.30423529, 0.2010557, 0.27523383, 0.42592946, -0.15043958,
               -0.08794977, -0.12733765, 0.22931154, -0.23913774, -0.0638661, -0.14958713,
               -0.04915145, 0.09898199, 0.05155884, -0.1142967, 0.16105883, 0.04871601,
               -0.08258422, -0.26105925]

    tpr, fpr, auc = get_roc(y_label, y_score)

    roc_plot(tpr, fpr, auc=auc)


if __name__ == "__main__":
    main()





