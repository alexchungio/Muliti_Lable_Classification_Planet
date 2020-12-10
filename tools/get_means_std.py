#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : get_means_std.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/10 下午4:10
# @ Software   : PyCharm
#-------------------------------------------------------

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from configs.cfgs import args


def get_mean_std():

    img_list = [img for img in os.listdir(args.train_image) if img[-4:] != '.csv']
    img_num = len(img_list)

    r_sum = g_sum = b_sum = 0
    mean_pbar = tqdm(enumerate(img_list, 0))
    for idx, img_name in mean_pbar:
        img_data = np.array(Image.open(os.path.join(args.train_image, img_name)).convert('RGB'))
        r_sum += np.sum(img_data[:, :, 0])
        g_sum += np.sum(img_data[:, :, 1])
        b_sum += np.sum(img_data[:, :, 2])
        mean_pbar.set_description(desc='calculating mean: {} / {}'.format(idx + 1, img_num))

    r_mean = r_sum / (256 * 256 * img_num)
    g_mean = g_sum / (256 * 256 * img_num)
    b_mean = b_sum / (256 * 256 * img_num)
    mean = (r_mean / 255, g_mean / 255, b_mean / 255)


    r_var_sum = g_var_sum = b_var_sum = 0
    std_pbar = tqdm(enumerate(img_list, 0))
    for idx, img_name in std_pbar:
        img_data = np.array(Image.open(os.path.join(args.train_image, img_name)).convert('RGB'))
        r_var_sum += np.sum(np.square(img_data[:, :, 0] - r_mean))
        g_var_sum += np.sum(np.square(img_data[:, :, 1] - g_mean))
        b_var_sum += np.sum(np.square(img_data[:, :, 2] - b_mean))
        std_pbar.set_description(desc='calculating std: {} / {}'.format(idx + 1, img_num),)

    r_std = np.sqrt(r_var_sum / (256 * 256 * img_num))
    g_std = np.sqrt(g_var_sum / (256 * 256 * img_num))
    b_std = np.sqrt(b_var_sum / (256 * 256 * img_num))
    std = (r_std / 255, g_std / 255, b_std / 255)

    return mean, std

def main():
    mean, std = get_mean_std()

    print ('r_mean: {:.3f}, g_mean: {:.3f}, b_mean: {:.3f}'.format(mean[0], mean[1], mean[2]))  # [0.311, 0.340, 0.299]
    print ('r_std: {:.3f}, g_std: {:.3f}, b_std: {:.3f}'.format(std[0], std[1], std[2]))  # [0.167, 0.144, 0.138]

if __name__ == "__main__":
    main()