#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : f_score.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/4 上午10:18
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix


def get_multi_confusion_matrix(y_label, y_pred):
    """

    :param y_label:
    :param y_pred:
    :return:
    """
    y_label = np.asarray(y_label, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    # get unique label
    labels = np.asarray(sorted(set(y_label)))
    # num classes
    num_labels = labels.size
    # label_to_ind = {y: x for x, y in enumerate(labels)}
    # # # convert yt, yp into index
    # y_pred = np.array([label_to_ind.get(x, num_labels + 1) for x in y_pred])
    # y_label = np.array([label_to_ind.get(x, num_labels + 1) for x in y_label])
    #-------------------------- get confusion matrix-----------------------------
    cm = np.zeros((num_labels, num_labels), dtype=np.int32)
    for l_index, p_index in zip(y_label, y_pred):
        cm[l_index, p_index] += 1

    # --------------------get multi confusion matrix----------------------------
    mul_cm = np.zeros((num_labels, 2, 2), dtype=np.int32)
    tp = np.diagonal(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    multi_cm = np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

    return cm, multi_cm


def get_precision_recall_f_score(y_true, y_pred, beta=1.0, average='micro'):

    cm, multi_cm = get_multi_confusion_matrix(y_true, y_pred)

    tp = multi_cm[:, 1, 1]
    fp = multi_cm[:, 0, 1]
    fn = multi_cm[:, 1, 0]

    tp_sum = tp
    pred_sum = tp + fp
    label_sum = tp + fn

    if average == 'micro':
        tp_sum = np.array([tp.sum()])
        pred_sum = np.array([pred_sum.sum()])
        label_sum = np.array([label_sum.sum()])

    precision = tp_sum / pred_sum
    recall = tp_sum / label_sum
    f1_score = (1+ beta **2) * precision * recall / ( beta **2 * precision + recall)

    precision = np.average(precision)
    recall = np.average(recall)
    f1_score = np.average(f1_score)

    return precision, recall, f1_score


def main():
    y_label = [0, 1, 2, 1, 1, 0, 2, 1, 0, 2]

    y_pred = [0, 1, 1, 2, 1, 0, 2, 0, 0, 2]


    # sk_multi_cm = multilabel_confusion_matrix(y_label, y_pred)
    cm, multi_cm = get_multi_confusion_matrix(y_label=y_label, y_pred=y_pred)

    # micro precision recall f1_score
    # micro_precision = precision_score(y_label, y_pred, average='micro')
    # micro_recall = recall_score(y_label, y_pred, average='micro')
    # micro_f1 = f1_score(y_label, y_pred, average='micro')
    micro_precision, micro_recall, micro_f1 = get_precision_recall_f_score(y_label, y_pred, average='micro')
    print(micro_precision, micro_recall, micro_f1)
    # macro precision recall f_score
    # macro_precision = precision_score(y_label, y_pred, average='micro')
    # macro_recall = recall_score(y_label, y_pred, average='micro')
    # macro_f1 = f1_score(y_label, y_pred, average='micro')
    macro_precision, macro_recall, macro_f1 = get_precision_recall_f_score(y_label, y_pred, average='macro')
    print(macro_precision, macro_recall, macro_f1)
    print('Done')


if __name__ == "__main__":
    main()