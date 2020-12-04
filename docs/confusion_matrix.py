#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : confusion_matrix.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/2 下午5:27
# @ Software   : PyCharm
#-------------------------------------------------------

import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def get_confusion_matrix(y_label, y_pred):
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
    cm = np.zeros((num_labels, num_labels), dtype=np.int32)

    for l_index, p_index in zip(y_label, y_pred):
        cm[l_index, p_index] += 1

    return cm

def visual_confusion_matrix(cm, normalize=False, title='confusion matrix', ticklabel=None):
    """

    :param cm:
    :param normalize:
    :param target_name:
    :return:
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    error = 1 - accuracy

    plt.figure(figsize=(8, 6))
    plt.title(title)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, center=0, linewidths=.5)

    if ticklabel is not None:
        tick_marks = np.arange(len(ticklabel)) + 0.5
        plt.xticks(tick_marks, ticklabel, rotation=45)
        plt.yticks(tick_marks, ticklabel)

    plt.tight_layout()
    plt.ylabel('label')
    plt.xlabel('predict \naccuracy={:0.4f} error={:0.4f}'.format(accuracy, error))
    plt.show()



def main():
    class_name = ['low', 'medium', 'high']

    y_label = [0, 1, 2, 1, 1, 0, 2, 1, 0, 2]

    y_pred =  [0, 1, 1, 2, 1, 0, 2, 0, 0, 2]

    # cm = confusion_matrix(y_label, y_pred)
    cm = get_confusion_matrix(y_label, y_pred)

    visual_confusion_matrix(cm, ticklabel=class_name)

if __name__ == "__main__":
    main()