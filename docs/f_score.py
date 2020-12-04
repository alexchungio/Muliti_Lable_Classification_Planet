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
import warnings

from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix


def get_confusion_matrix(y_label, y_pred):

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
    # -------------------------- get confusion matrix-----------------------------
    cm = np.zeros((num_labels, num_labels), dtype=np.int32)
    for l_index, p_index in zip(y_label, y_pred):
        cm[l_index, p_index] += 1

    return cm


def get_multi_class_confusion_matrix(y_label, y_pred):
    """

    :param y_label:
    :param y_pred:
    :return:
    """
    cm = get_confusion_matrix(y_label, y_pred)
    # --------------------get multi confusion matrix----------------------------
    tp = np.diagonal(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    multi_cm = np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

    return cm, multi_cm


def single_label_precision_recall_f(y_true, y_pred, beta=1.0, average='micro'):

    # ----------------------get confusion matrix of per class------------------------------
    cm, multi_cm = get_multi_class_confusion_matrix(y_true, y_pred)

    # ----------------------computer precision recall and f-score-------------------------
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

    # removing warnings if zero_division is set to something different than its default value
    warnings.filterwarnings("ignore")

    precision = tp_sum / pred_sum
    recall = tp_sum / label_sum
    f1_score = (1+ beta **2) * precision * recall / ( beta **2 * precision + recall)
    f1_score[np.isnan(f1_score)] = 0

    precision = np.average(precision)
    recall = np.average(recall)
    f1_score = np.average(f1_score)

    return precision, recall, f1_score


def multi_label_precision_recall_f(y_label, y_pred, beta=1.0, average='micro'):
    """

    :param y_label:
    :param y_pred:
    :param beta:
    :return:
    """
    y_label = np.asarray(y_label, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    assert y_label.shape == y_pred.shape

    # ----------------------get confusion matrix of per class------------------------------
    num_class = y_label.shape[1]

    cms = []
    multi_cms = np.zeros((0, 2, 2))

    for i in range(num_class):
        cm, multi_cm = get_multi_class_confusion_matrix(y_label[:, i], y_pred[:, i])
        cms.append(cm)
        multi_cms = np.concatenate([multi_cms, multi_cm[1][np.newaxis, :]])


    # ----------------------computer precision recall and f-score-------------------------
    tp = multi_cms[:, 1, 1]
    fp = multi_cms[:, 0, 1]
    fn = multi_cms[:, 1, 0]

    tp_sum = tp
    pred_sum = tp + fp
    label_sum = tp + fn

    if average == 'micro':
        tp_sum = np.array([tp.sum()])
        pred_sum = np.array([pred_sum.sum()])
        label_sum = np.array([label_sum.sum()])

    precision = tp_sum / pred_sum

    # removing warnings if zero_division is set to something different than its default value
    warnings.filterwarnings('ignore')

    recall = tp_sum / label_sum
    f1_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    f1_score[np.isnan(f1_score)] = 0

    precision = np.average(precision)
    recall = np.average(recall)
    f1_score = np.average(f1_score)

    return precision, recall, f1_score


def main():

    # binary class classify
    binary_class_true = [0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
    binary_class_pred = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

    # multi class classify
    multi_class_true = [0, 1, 2, 1, 1, 0, 2, 1, 0, 2]
    multi_class_pred = [0, 1, 1, 2, 1, 0, 2, 0, 0, 2]

    # multi label classify
    multi_label_true = [[1, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0],
                        [0, 1, 1, 0, 1]]

    multi_label_pred = [[1, 0, 0, 1, 1],
                        [1, 0, 1, 1, 0],
                       [0, 1, 0, 0, 1]]


    y_label = binary_class_true
    y_pred = binary_class_pred
    #
    # # sk_multi_cm = multilabel_confusion_matrix(y_label, y_pred)
    # cm, multi_cm = get_multi_confusion_matrix(y_label=y_label, y_pred=y_pred)
    #
    # micro precision recall f1_score
    # micro_precision = precision_score(y_label, y_pred, average='micro')
    # micro_recall = recall_score(y_label, y_pred, average='micro')
    # micro_f1 = f1_score(y_label, y_pred, average='micro')
    micro_precision, micro_recall, micro_f1 = single_label_precision_recall_f(y_label, y_pred, average='micro')
    print(micro_precision, micro_recall, micro_f1)
    # macro precision recall f_score
    # macro_precision = precision_score(y_label, y_pred, average='micro')
    # macro_recall = recall_score(y_label, y_pred, average='micro')
    # macro_f1 = f1_score(y_label, y_pred, average='micro')
    macro_precision, macro_recall, macro_f1 = single_label_precision_recall_f(y_label, y_pred, average='macro')
    print(macro_precision, macro_recall, macro_f1)
    print('Done')


    # micro_precision = precision_score(multi_label_true, multi_label_pred, average="micro")  # 0.6
    # macro_precision = recall_score(multi_label_true, multi_label_pred, average="micro")
    # micro_f1 = f1_score(multi_label_true, multi_label_pred, average="micro")

    micro_precision, macro_precision, micro_f1 = multi_label_precision_recall_f(multi_label_true, multi_label_pred)


    print(micro_precision, macro_precision, micro_f1)




if __name__ == "__main__":
    main()