#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : transform.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/10 下午3:03
# @ Software   : PyCharm
#-------------------------------------------------------
import random
from PIL import Image, ImageFilter
import torch
from torchvision.transforms import transforms

from data.dataset import PlanetDataset
from configs.cfgs import args


class AspectPreservingResize(object):
    def __init__(self, smallest_side, interpolation=Image.BILINEAR):

        """

        :param smallest_side: int
        :param interpolation:
        """
        self.smallest_size = smallest_side
        self.interpolation = interpolation

    def __call__(self, img):

        w, h = img.size
        scale = self.smallest_size / w if w < h else self.smallest_size / h
        w_target, h_target = int(w * scale), int(h * scale)
        img = img.resize((w_target, h_target), self.interpolation)

        return img


class RelativePreservingResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        """

        :param size:  width, height
        :param interpolation:
        """

        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size

        if w / h < ratio:  # padding width
            # target width
            w_target = h * ratio
            w_padding = (w_target - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else: # padding height
            h_target = int(w / ratio)
            h_padding = (h_target - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


class RandomShift(object):
    def __init__(self, p=0.25, size=4):
        self.p = p
        self.size = size

    def __call__(self, image):
        h, w = image.size
        if random.random() < self.p:
            dx = round(random.uniform(-self.size, self.size))  # pixel
            dy = round(random.uniform(-self.size, self.size))  # pixel

            x_0 = 0 - dx
            y_0 = 0 - dy

            x_1 = x_0 + w
            y_1 = y_0 + h
            image = image.crop((y_0, x_0, y_1, x_1))

        return image


class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img


class RandomSharpe(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.SHARPEN())
        return img


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomFilter(RandomOrder):

    def __init__(self, blur=0, sharp=None):
        self.transforms = []
        if blur is not None:
            self.transforms.append(RandomGaussianBlur())
        if sharp is not None:
            self.transforms.append(RandomSharpe())
        super(RandomFilter, self).__init__(transforms=self.transforms)


def get_train_transform(mean, std, size):
    """
    Data augmentation and normalization for training
    :param mean:
    :param std:
    :param size: width, height
    :return:
    """
    if isinstance(size, int):
        size = (int(size), (size))
    else:
        size = size

    train_transform = transforms.Compose([
        RelativePreservingResize((int(size[0] * (256 / 224)), int(size[1] * (256 / 224)))),
        transforms.RandomCrop(size),
        RandomShift(p=0.25, size=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        RandomFilter(blur=True),
        RandomRotate(90, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_test_transform(mean, std, size):
    """
    Just normalization for validation
    :param mean:
    :param std:
    :param size: width, height
    :return:
    """
    if isinstance(size, int):
        size = (int(size), (size))
    else:
        size = size

    test_transform = transforms.Compose([
        RelativePreservingResize((int(size[0] * (256 / 224)), int(size[1] * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return test_transform


def get_transform(size, mode='test'):

    assert mode in ['train', 'val', 'test']

    if not isinstance(size, tuple):
        size = (size, size)

    mean =  [0.311, 0.340, 0.299]
    std = [0.167, 0.144, 0.138]
    if mode in ['train']:
        transform =get_train_transform(mean, std, size)
    else:
        transform = get_test_transform(mean, std, size)

    return transform



def main():

    # train_transform = get_transform(size=256, mode='train')

    train_dataset = PlanetDataset(image_root=args.train_image, target_path=args.labels)

    blur = RandomFilter(blur=True)
    shift = RandomShift()
    for image, label in train_dataset:
        blur_image = blur(image)
        shift_image = shift(blur_image)

        shift_image.show()

        break


if __name__ == "__main__":
    main()