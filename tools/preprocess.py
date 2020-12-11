#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : preprocess.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/10 上午9:31
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import os
import pandas as pd
import numpy as np
import csv
import math
from collections import Counter

from configs.cfgs import  args
from utils.misc import read_class_names

def class_reweight(class_count):
    """
    re-sample | re-weight
    use training long tail dataset
    :param class_count:
    :return:
    """
    assert isinstance(class_count, dict)
    # sort
    order_class_count = sorted(class_count.items(), key=lambda x:x[1], reverse=False)

    order_class = [name for name, count in order_class_count]
    order_count = np.array([count for name, count in order_class_count], dtype=np.int32)

    # get weight
    class_weight = np.log(np.sum(order_count) / order_count)
    # minimize weight equal to 1
    class_weight /= class_weight[-1]

    with open(args.classes, 'w') as fw:
        for c in order_class:
            fw.write('{}\n'.format(c))

    with open(args.class_weights, 'w')as fw:
        for w in class_weight:
            fw.write('{}\n'.format(w))


def count_class():
    # convert train tag

    train_df = pd.read_csv(args.train_label)
    train_df.tags = train_df.tags.map(lambda x: set(x.split()))

    # get number of tag
    count = Counter()
    train_df.tags.apply(lambda x: count.update(x))
    # get class reweight
    class_reweight(dict(count))
    # convert label to index
    for k in count:
        train_df[k] = [1 if k in tag else 0 for tag in train_df.tags]

    # train_df = train_df[(train_df[LABEL_SKY_COVER].T != 0).any()]
    os.makedirs(os.path.dirname(args.tags_count), exist_ok=True)
    with open(args.tags_count, 'w') as f:
        w = csv.writer(f)
        w.writerow(['tag', 'count'])
        for k, v in count.items():
            print(k, v)
            w.writerow([k, v])
        f.close()

    os.makedirs(os.path.dirname(args.corr), exist_ok=True)
    tags_only = train_df[list(count.keys())]
    corr = tags_only.corr()
    corr.to_csv(args.corr)

    return train_df, count

def get_fold_data(train_df, count, num_folds=5):
    """

    :param train_df:
    :param count:
    :param num_folds:
    :return:
    """
    attempt = 0
    target_counts = {k: (v / num_folds) for k, v in count.items()}
    target_thresh = {k: max(1., v * .20) for k, v in target_counts.items()}
    print(target_counts, target_thresh)

    furthest_fold = 0
    fold_counts = []
    while attempt < 1000000:
        train_df['fold'] = np.random.randint(0, num_folds, size=len(train_df.index))
        valid = True
        ss = train_df.groupby('fold').sum()
        for f in range(num_folds):
            sr = ss.iloc[f]
            fold_counts.append(sr)
            for k, v in sr.items():
                target = target_counts[k]
                thresh = target_thresh[k]
                diff = math.floor(abs(v - target))
                thresh = math.ceil(thresh)
                if diff > thresh:
                    valid = False
                    if f > furthest_fold:
                        furthest_fold = f
                        print(f, abs(v - target), math.ceil(thresh), k)
                    break
            if not valid:
                break
        if valid:
            break
        else:
            fold_counts = []
            attempt += 1
    print(attempt, furthest_fold)
    for i, x in enumerate(fold_counts):
        print(i)
        for k, v in x.items():
            print(k, v)
    labels_df = train_df[['image_name', 'fold'] + list(count.keys())]
    labels_df.to_csv(args.labels, index=False)


def main():
    train_label = args.train_label
    print(train_label)
    # class_name = read_class_names(args.classes)

    train_df, count = count_class()
    get_fold_data(train_df, count)
    print('Done')


if __name__ == "__main__":
    main()