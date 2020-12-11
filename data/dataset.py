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
import re

from data.transform import get_transform
from utils.misc import plt_imshow, read_class_names, read_class_weights, index_to_tag

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
LABEL_TYPE = ['all']

NAME_INDEX, INDEX_NAME = read_class_names(args.classes)

# ALL_WEIGHTS =
ALL_WEIGHTS_L = read_class_weights(args.class_weights)


def get_tags(tags_type='all'):

    if tags_type == 'all':
        return NAME_INDEX
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
            image = self.transform(image)

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

    train_transform = get_transform(size=256, mode='train')
    train_dataset = PlanetDataset(image_root=args.train_image, target_path=args.labels, transform=train_transform)

    for image, label in train_dataset:

        tags = index_to_tag(label, index_tag=INDEX_NAME)
        plt_imshow(image, title=tags)

        break
if __name__ == "__main__":
    main()
