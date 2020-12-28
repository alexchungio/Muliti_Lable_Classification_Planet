#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/30 上午11:02
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import random
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.transforms import transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from configs.cfgs import args
from data.dataset import PlanetDataset
from data.transform import get_transform
from models.model_factory import build_model
from utils.tools import get_optimizer, AverageMeter, accuracy


#--------------------------------global config-----------------------------------
# set random sees
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)

random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.benchmark = True

# set writer
writer = SummaryWriter(log_dir=args.summary)

# global step
global_step = 0

def main():
    # --------------------------------config-------------------------------
    global use_cuda
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    # ------------------------------ load dataset---------------------------
    print('==> Loader dataset {}'.format(args.train_data))

    train_transform = get_transform(size=args.image_size, mode='train')
    train_dataset = PlanetDataset(image_root=args.train_data,
                                  target_path=args.labels,
                                  phase='train',
                                  fold=args.fold,
                                  img_type=args.image_type,
                                  img_size = args.image_size,
                                  transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_works)

    eval_transform = get_transform(size=args.image_size, mode='eval')
    eval_dataset = PlanetDataset(image_root=args.train_data,
                                  target_path=args.labels,
                                  phase='eval',
                                  fold=args.fold,
                                  img_type=args.image_type,
                                  img_size=args.image_size,
                                  transform=eval_transform)
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_works)


    # ---------------------------------model---------------------------------
    model = build_model(model_name=args.model_name, num_classes=args.num_classes, pretrained=args.pretrained,
                        global_pool=args.global_pool)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()  # load model to cuda
    # show model size
    print('\t Total params volumes: {:.2f} M'.format(sum(param.numel() for param in model.parameters()) / 1000000.0))

    # --------------------------------criterion-----------------------
    if args.class_weights:
        class_weights = torch.from_numpy(train_dataset.get_class_weights()).float()
        class_weights_norm = class_weights / class_weights.sum()
        if use_cuda:
            class_weights = class_weights.cuda()
            class_weights_norm = class_weights_norm.cuda()
    else:
        class_weights = None
        class_weights_norm = None

    if args.loss.lower() == 'nll':
        # assert not args.multi_label and 'Cannot use crossentropy with multi-label target.'
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss.lower() == 'mlsm':
        assert args.multi_label
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    else:
        assert False and "Invalid loss function"


    #---------------------------------optimizer----------------------------
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)


    # # Resume model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(-1)


    # eval model
    # if args.evaluate:
    #     print('\nEvaluation only')
    #     test_loss, test_acc_1, test_acc_5 = test(val_loader, model, criterion, use_cuda)
    #     print(' Test => loss {:.4f} | acc_top1 {:.4f} acc_top5'.format(test_loss, test_acc_1, test_acc_5))
    #
    #     return None
    #
    # # best_model_weights = copy.deepcopy(model.state_dict())
    # since = time.time()
    for epoch in range(start_epoch, args.epochs):
        # print('Epoch {}/{} | LR {:.8f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss = train(loader=train_loader, model=model, epoch=epoch, criterion=criterion, optimizer=optimizer,
                           summary_iter = args.summary_iter, class_weights=class_weights, use_cuda=use_cuda)
        # test_loss, test_acc_1, test_acc_5 = test(val_loader, model, criterion, use_cuda)
        #
        # scheduler.step(metrics=test_loss)
        #
        # # save logs
        # writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train': train_loss, 'val': test_loss},
        #                    global_step=epoch)
        # writer.add_scalars(main_tag='epoch/acc_top1', tag_scalar_dict={'train': train_acc_1, 'val': test_acc_1},
        #                    global_step=epoch)
        # writer.add_scalars(main_tag='epoch/acc_top5', tag_scalar_dict={'train': train_acc_5, 'val': test_acc_5},
        #                    global_step=epoch)
        #
        # # add learning_rate to logs
        # writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
    #
    #     # -----------------------------save model every epoch -----------------------------
    #     # get param state dict
    #     if len(args.gpu_id) > 1:
    #         best_model_weights = model.module.state_dict()
    #     else:
    #         best_model_weights = model.state_dict()
    #
    #     state = {
    #         'epoch': epoch + 1,
    #         'acc': best_acc,
    #         'state_dict': best_model_weights,
    #         'optimizer': optimizer.state_dict()
    #     }
    #     save_checkpoint(state, args.checkpoint)
    #
    #     # save best checkpoint
    #     if test_acc_1 > best_acc:
    #         best_acc = test_acc_1
    #         shutil.copy(args.checkpoint, args.best_checkpoint)
    #
    # writer.close()
    #
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

def train(loader, model, epoch, criterion,  optimizer, summary_iter, class_weights=None, use_cuda=None):

    global global_step
    model.train()
    losses = AverageMeter()
    # acc_top1 = AverageMeter()
    # acc_top5 = AverageMeter()
    batch_time = AverageMeter()

    start_time = time.time()
    pbar = tqdm(loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        input_var = torch.autograd.Variable(inputs)

        if args.multi_label and args.loss == 'null':
            # if multi-label and null set, train network by sampling an index using class weights
            if class_weights is not None:
                target_weights = targets * torch.unsqueeze(class_weights, 0).expand_as(targets)
                sum_weights = target_weights.sum(dim=1, keepdim=True).expand_as(target_weights)
                target_weights = target_weights.div(sum_weights)
            else:
                target_weights = targets
            target_var = torch.autograd.Variable(torch.multinomial(target_weights, 1).squeeze().long())
        else:
            target_var = torch.autograd.Variable(targets)

        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.item(), input_var.size(0))

        # measure accuracy and record
        # acc_1, acc_5 = accuracy(outputs.data, target=targets.data, topk=(1, 5))
        # losses.update(loss.item(), inputs.size(0))
        # acc_top1.update(acc_1.item(), inputs.size(0))
        # acc_top5.update(acc_5.item(), inputs.size(0))
        #
        # if (global_step + 1) % summary_iter == 0:
        #     writer.add_scalar(tag='train/loss', scalar_value=loss.cpu().item(), global_step=global_step)
        #     writer.add_scalar(tag='train/acc_top1', scalar_value= acc_1, global_step=global_step)
        #     writer.add_scalar(tag='train/acc_top5', scalar_value=acc_5, global_step=global_step)

        # grad clearing
        optimizer.zero_grad()
        # computer grad
        loss.backward()
        # update params
        optimizer.step()

        global_step += 1

        # pbar.set_description('train loss {0}'.format(loss.item()), refresh=False)

        batch_time.update(time.time() - start_time)

        pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                                        batch_idx * len(input_var),
                                                                        len(loader.sampler),
                                                                        100. * batch_idx / len(loader)))
        pbar.set_postfix_str('Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                         'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                         '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '.format(
                         loss=losses,
                         batch_time=batch_time,
                         rate=input_var.size(0) / batch_time.val,
                         rate_avg=input_var.size(0) / batch_time.avg), refresh=True)

        start_time = time.time()


    # pbar.write()

    return OrderedDict([('loss', losses.avg)])


def test(eval_loader, model, criterion, use_cuda):

    model.eval()

    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    pbar = tqdm(eval_loader)
    with torch.set_grad_enabled(mode=False):
        for inputs, targets in pbar:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc_1, acc_5 = accuracy(outputs.data, target=targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc_top1.update(acc_1.item(), inputs.size(0))
            acc_top5.update(acc_5.item(), inputs.size(0))

            pbar.set_description('eval loss {0}'.format(loss.item()), refresh=False)

    pbar.write('\teval => loss {:.4f} | acc_top1 {:.4f}  acc_top5 {:.4f}'.format(losses.avg, acc_top1.avg, acc_top5.avg))

    return (losses.avg, acc_top1.avg, acc_top5.avg)




if __name__ == "__main__":
    main()







