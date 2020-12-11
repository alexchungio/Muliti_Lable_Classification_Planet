#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : misc.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/10 上午10:07
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

_all__ = ['read_class_names']


def read_class_names(class_file_name):
    '''loads class name from a file'''
    name_index, index_name = {}, {}
    with open(class_file_name, 'r') as data:
        for id, name in enumerate(data):
            index_name[id] = name.strip('\n')
            name_index[name.strip('\n')] = id
    return name_index, index_name

def read_class_weights(weight_name):
    '''loads class name from a file'''
    weights = list()
    with open(weight_name, 'r') as data:
        for id, weight in enumerate(data):
            weights.append(float(weight.strip('\n')))
    return weights


def plt_imshow(image, title=None):
    """

    :param image:
    :param title:
    :return:
    """
    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.311, 0.340, 0.299])
    std = np.array([0.167, 0.144, 0.138])
    image = std * image + mean
    image = np.clip(image, 0, 1)  # clip to (0, 1)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


def index_to_tag(v, index_tag):
    """

    :param v: vector
    :param index_tag:
    :return:
    """
    v = v.numpy()
    idx = np.nonzero(v)
    tags = [index_tag[i] for i in idx[0]]

    return ' '.join(tags)