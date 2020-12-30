#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : ensemble.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/11 上午11:02
# @ Software   : PyCharm
#-------------------------------------------------------

import argparse
import os
import time
import numpy as np
import pandas as pd


from configs.cfgs import args
from utils.misc import read_class_names, index_to_tag



submission_col = ['image_name', 'tags']

name_index, index_name = read_class_names(args.classes)


def find_inputs(folder, types=['.csv']):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs



def main():

    subs = find_inputs(args.inference_path, types=['.csv'])
    dfs = []
    for s in subs:

        df = pd.read_csv(s[1], index_col=None)
        if 'tags' in df.columns:
            df = df.set_index('image_name')
            df.tags = df.tags.map(lambda x: set(x.split()))
            for l in name_index.keys():
                df[l] = [1 if l in tag else 0 for tag in df.tags]
            df.drop(['tags'], inplace=True, axis=1)
            dfs.append(df)
        else:
            pass

    assert len(dfs)

    d = dfs[0]
    for o in dfs[1:]:
        d = d.add(o)
    d = d / len(dfs)
    b = (d >= 0.42).astype(int)

    m = b.iloc[:, :].values
    out = []
    for i, x in enumerate(m):
        t = index_to_tag(x, index_name)
        out.append([b.index[i]] + [t])

    results_sub_df = pd.DataFrame(out, columns=submission_col)
    results_sub_df.to_csv(os.path.join(args.inference_path, 'submission-e.csv'), index=False)


if __name__ == '__main__':
    main()
