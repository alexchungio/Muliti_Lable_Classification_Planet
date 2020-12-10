#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/10 下午3:03
# @ Software   : PyCharm
#-------------------------------------------------------

import torch.utils.data as data

from configs.cfgs import args


import cv2
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import numpy as np
import math
import os
import utils
import re

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
LABEL_TYPE = ['all']

LABEL_ALL = [
    'blow_down',
    'conventional_mine',
    'slash_burn',
    'blooming',
    'artisinal_mine',
    'selective_logging',
    'bare_ground',
    'cloudy',
    'haze',
    'habitation',
    'cultivation',
    'partly_cloudy',
    'water',
    'road',
    'agriculture',
    'clear',
    'primary',
]

ALL_WEIGHTS = [
    382.7857142857,
    375.13,
    179.4880382775,
    112.9909638554,
    110.6578171091,
    110.3323529412,
    43.5185614849,
    17.9573958832,
    13.9091583241,
    10.2494535519,
    8.37904847,
    5.1663682688,
    5.061800027,
    4.6478751084,
    3.0461226147,
    1.3194400478,
    1.,
]

ALL_WEIGHTS_L = [
    8.5841572006,
    8.5550875696,
    7.4957594164,
    6.8327756552,
    6.8029404468,
    6.7987290874,
    5.4763350709,
    4.2446888943,
    3.8981269095,
    3.4917830184,
    3.2294415648,
    2.6244210529,
    2.5997462598,
    2.4977081866,
    2.0165400403,
    1.2137765563,
    1
]



LABEL_PRIMARY = ['primary']


def get_tags(tags_type='all'):

    if tags_type == 'all':
        return LABEL_ALL
    else:
        assert False and "Invalid label type"
        return []


def get_tags_size(tags_type='all'):
    return len(get_tags(tags_type))


def get_class_weights(tags_type='all'):
    if tags_type == 'all':
        return np.array(ALL_WEIGHTS_L)
    else:
        return np.array([])

def get_valid_image(folder, types=None):
    """

    :param folder:
    :param types:
    :return:
    """
    images = {}
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base_name, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_path = os.path.join(root, rel_filename)
                images[base_name] = abs_path
    return images


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


class PlanetDataset(data.Dataset):
    def __init__(
            self,
            image_root,
            target_path='',
            tags_type='all',
            multi_label=True,
            train=True,
            fold=0,
            img_type='.jpg',
            img_size=(256, 256),
            transform=None):
        super(PlanetDataset, self).__init__()

        self.tags_type = tags_type
        self.image_root = image_root
        self.target_path = target_path
        self.img_size = img_size
        self.transform = transform

        # -------------------get valid images------------------------
        assert img_type in ['.jpg', '.tif']
        images = get_valid_image(image_root, types=[img_type])
        if len(images) == 0:
            raise (RuntimeError("Found 0 images in : " + image_root))

        # ------------------get target-------------------------------
        # train
        if target_path:
            target_df = pd.read_csv(target_path)

            # get data by train or eval
            if train:
                target_df = target_df[target_df['fold'] != fold]
            else:
                target_df = target_df[target_df['fold'] == fold]
            target_df.drop(['fold'], 1, inplace=True)

            print(len(images), len(target_df.index))

            target_df = target_df[target_df.image_name.map(lambda x: x in images)]
            target_df['filename'] = target_df.image_name.map(lambda x: images[x])
            self.images = target_df['filename'].tolist()


            tags = get_tags(tags_type)
            self.target_array = target_df[tags].values.astype(np.float32)

            if not multi_label:
                self.target_array = np.argmax(self.target_array, axis=1)
            self.target_array = torch.from_numpy(self.target_array)
        # test / inference
        else:
            assert not train
            self.images = sorted(images, key=lambda x: natural_key(x[0]))
            self.target_array = None


    def __getitem__(self, index):


        image = self.load_image(self.images[index]).convert('RGB')


        label = self.target_array[index]

        if self.transform is not None:
            image = self.transform(image, size=self.img_size)

        return image, label


    def load_image(self, path):
        """

        :param path:
        :return:
        """
        image = Image.open(path)
        return image.convert('RGB')


    def __len__(self):

        return len(self.images)

    def get_class_weights(self):

        return get_class_weights(self.tags_type)

    def get_sample_weights(self):

        class_weights = torch.FloatTensor(self.get_class_weights())
        weighted_samples = []
        for index in range(len(self.images)):
            masked_weights = self.target_array[index] * class_weights
            weighted_samples.append(masked_weights.max())
        weighted_samples = torch.DoubleTensor(weighted_samples)
        weighted_samples = weighted_samples / weighted_samples.min()
        return weighted_samples


def main():
    train_dataset = PlanetDataset(image_root=args.train_image, target_path=args.labels)

    for image, label in train_dataset:
        image.show()
        print(label)
        break

if __name__ == "__main__":
    main()

