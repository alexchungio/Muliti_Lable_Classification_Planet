#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
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
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torch.nn.functional as F

from configs.cfgs import args
from data.dataset import PlanetDataset
from data.transform import get_transform
from models.model_factory import build_model
from utils.tools import AverageMeter
from utils.misc import read_class_names, index_to_tag


# config gpu
use_cuda = torch.cuda.is_available()
gpu_ids = list(map(int, args.gpu_id.split(',')))

if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.benchmark = True


def inference(data_path, threshold=0.3):

    name_index, index_name = read_class_names(args.classes)
    output_col = ['image_name'] + list(name_index.keys())
    submission_col = ['image_name', 'tags']
    inference_dir = None

    #----------------------------dataset generator--------------------------------
    test_transform = get_transform(size=args.image_size, mode='test')
    test_dataset = PlanetDataset(image_root=data_path,
                                 phase='test',
                                 img_type=args.image_type,
                                 img_size=args.image_size,
                                 transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_works)
    # ---------------------------------load model and param--------------------------------
    model = build_model(model_name = args.model_name, num_classes=args.num_classes, global_pool=args.global_pool)

    if args.best_checkpoint is not None:
        assert os.path.isfile(args.best_checkpoint), '{} not found'.format(args.best_checkpoint)
        checkpoint = torch.load(args.best_checkpoint)
        print('Restoring model with {} architecture...'.format(checkpoint['arch']))

        # load model weights
        if use_cuda:
            if checkpoint['num_gpu'] > 1:
                model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
            else:
                model.cuda()
        else:
            if checkpoint['num_gpu'] > 1:
                model = torch.nn.DataParallel(model)
            else:
                model.cuda()
        model.load_state_dict(checkpoint['state_dict'])

        # update threshold
        if 'threshold' in checkpoint:
            threshold = checkpoint['threshold']
            threshold = torch.tensor(threshold, dtype=torch.float32)
            print('Using thresholds:', threshold)
        else:
            threshold = 0.3

        if use_cuda:
            threshold = threshold.cuda()

        # generate save path
        inference_dir = os.path.join(os.path.normcase(args.inference_path), '{}-f{}-{:.6f}'.
                                     format(checkpoint['arch'], checkpoint['fold'], checkpoint['f2']))
        os.makedirs(inference_dir, exist_ok=True)

        print('Model restored from file: {}'.format(args.best_checkpoint))
    else:
        assert False and "No checkpoint specified"

    # -------------------------------------inference---------------------------------------
    model.eval()

    batch_time_meter = AverageMeter()
    results_raw = []
    results_label = []
    results_submission = []

    since_time = time.time()
    pbar = tqdm(enumerate(test_loader))
    try:
        with torch.no_grad():
            start = time.time()
            for batch_idx, (inputs, _, indices) in pbar:
                if use_cuda:
                    inputs = inputs.cuda()
                # input_var = autograd.Variable(input, volatile=True)
                input_var = torch.autograd.Variable(inputs)
                outputs = model(input_var)

                if args.multi_label:
                    if args.loss == 'nll':
                        outputs = F.softmax(outputs)
                    else:
                        outputs = torch.sigmoid(outputs)

                expand_threshold = torch.unsqueeze(threshold, 0).expand_as(outputs)
                output_labels = (outputs.data > expand_threshold).byte()

                # move data to CPU and collect
                outputs = outputs.cpu().data.numpy()
                output_labels = output_labels.cpu().numpy()
                indices = indices.cpu().numpy().flatten()

                for index, output, output_label in zip(indices, outputs, output_labels):

                    image_name = os.path.splitext(os.path.basename(test_dataset.images[index]))[0]
                    results_raw.append([image_name] + list(output))
                    results_label.append([image_name] + list(output_label))
                    results_submission.append([image_name] + [index_to_tag(output_label, index_name)])

                batch_time_meter.update(time.time() - start)
                if batch_idx % args.summary_iter == 0:
                    print('Inference: [{}/{} ({:.0f}%)]  '
                          'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                          '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '.format(
                        batch_idx * len(inputs), len(test_loader.sampler), 100. * batch_idx / len(test_loader),
                        batch_time=batch_time_meter,
                        rate=input_var.size(0) / batch_time_meter.val,
                        rate_avg=input_var.size(0) / batch_time_meter.avg))

                start = time.time()

    except KeyboardInterrupt:
        pass

    results_raw_df = pd.DataFrame(results_raw, columns=output_col)
    results_raw_df.to_csv(os.path.join(inference_dir, 'results_raw.csv'), index=False)
    results_label_df = pd.DataFrame(results_label, columns=output_col)
    results_label_df.to_csv(os.path.join(inference_dir, 'results_thr.csv'), index=False)
    results_submission_df = pd.DataFrame(results_submission, columns=submission_col)
    results_submission_df.to_csv(os.path.join(inference_dir, 'submission.csv'), index=False)

    time_elapsed = time.time() - since_time
    print('*** Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def main():

    test_data = os.path.join(args.dataset, 'train-jpg')

    # print(os.path.normpath(test_data))

    inference(data_path=test_data)


if __name__ == '__main__':
    main()
