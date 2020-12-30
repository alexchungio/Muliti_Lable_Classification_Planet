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
from utils.tools import get_optimizer, AverageMeter, accuracy, scores, f2_score, optimise_f2_thresholds, save_checkpoint


#--------------------------------global config-----------------------------------
# set random sees
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)

random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
gpu_ids = list(map(int, args.gpu_id.split(',')))

if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.benchmark = True

# set writer
writer = SummaryWriter(log_dir=args.summary_dir)

# global step
global_step = 0

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

def main():
    # --------------------------------config-------------------------------
    global use_cuda
    global gpu_ids
    threshold = args.threshold
    best_loss = None
    best_f2 = None

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
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_works)


    # ---------------------------------model---------------------------------
    model = build_model(model_name=args.model_name, num_classes=args.num_classes, pretrained=args.pretrained,
                        global_pool=args.global_pool)
    if use_cuda:
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()  # load model to cuda
        else:
            model.cuda()
    # show model size
    print('\t Total params volumes: {:.2f} M'.format(sum(param.numel() for param in model.parameters()) / 1000000.0))

    # --------------------------------criterion-----------------------
    criterion = None
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

    # lr scheduler
    if not args.decay_epoch:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,                                                          patience=8, verbose=False)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.1)

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
    # best_model_weights = copy.deepcopy(model.state_dict())
    since = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            # print('Epoch {}/{} | LR {:.8f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr']))

            train_metrics = train(loader=train_loader, model=model, epoch=epoch, criterion=criterion, optimizer=optimizer,
                               threshold=threshold, class_weights=class_weights_norm, use_cuda=use_cuda)
            eval_metrics, latest_threshold = eval(loader=eval_loader, model=model, epoch=epoch, criterion=criterion,
                                                  threshold=threshold, use_cuda=use_cuda)

            if args.decay_epoch is None:
                lr_scheduler.step(eval_metrics['loss'])
            else:
                lr_scheduler.step()

            # save train and eval metric
            writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train': train_metrics['loss'],
                                                                       'val': eval_metrics['loss']},
                               global_step=epoch)

            if args.multi_label:
                writer.add_scalars(main_tag='epoch/acc', tag_scalar_dict={'train': train_metrics['acc'],
                                                                           'val': eval_metrics['acc']},
                                   global_step=epoch)
            else:
                writer.add_scalars(main_tag='epoch/acc_top1', tag_scalar_dict={'train': train_metrics['acc_top1'],
                                                                          'val': eval_metrics['acc_top1']},
                                   global_step=epoch)
                writer.add_scalars(main_tag='epoch/acc_top5', tag_scalar_dict={'train': train_metrics['acc_top5'],
                                                                               'val': eval_metrics['acc_top5']},
                                   global_step=epoch)

            writer.add_scalar(tag='epoch/f2_score', scalar_value=eval_metrics['f2'],
                               global_step=epoch)

            # add learning_rate to logs
            writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)

            # -----------------------------save model every epoch -----------------------------
            # get param state dict
            if len(args.gpu_id) > 1:
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()

            # -------------------------- save model state--------------------------
            is_best = False

            if best_loss is not None or best_f2 is not None:
                if eval_metrics['loss'] < best_loss[0]:
                    best_loss = (eval_metrics['loss'], epoch)
                    if args.score_metric == 'loss':
                        is_best = True
                elif  eval_metrics['f2'] > best_f2[0]:
                    best_f2 = (eval_metrics['f2'], epoch)
                    if args.score_metric == 'f2':
                        is_best = True
                else:
                    is_best = False
                    pass
            else:
                best_loss = (eval_metrics['loss'], epoch)
                best_f2 = (eval_metrics['f2'], epoch)
                is_best = True


            state = {
                'epoch': epoch + 1,
                'arch': args.model_name,
                'state_dict': model_weights,
                'optimizer': optimizer.state_dict(),
                'threshold': latest_threshold,
                'loss': eval_metrics['loss'],
                'f2': eval_metrics['f2'],
                'fold': args.fold,
                'num_gpu': len(gpu_ids)
            }
            save_checkpoint(state,
                            os.path.join(args.checkpoint, 'ckpt-{}-f{}-{:.6f}.pth.tar'.format(epoch, args.fold,
                                                                                              eval_metrics['f2'])),
                            is_best=is_best)

    except KeyboardInterrupt:
        pass

    writer.close()

    time_elapsed = time.time() - since
    print('*** Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('*** Eval best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))
    print('*** Eval best f2_score: {0} (epoch {1})'.format(best_f2[1], best_f2[0]))


def train(loader, model, epoch, criterion,  optimizer, threshold, class_weights=None, use_cuda=None):
    """

    :param loader:
    :param model:
    :param epoch:
    :param criterion:
    :param optimizer:
    :param threshold:
    :param class_weights:
    :param use_cuda:
    :return:
    """
    model.train()

    global global_step

    acc_top1_meter = AverageMeter()
    acc_top5_meter = AverageMeter()

    precision_meter = AverageMeter()
    acc_meter = AverageMeter()
    f2_score_meter = AverageMeter()

    loss_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    start_time = time.time()
    pbar = tqdm(loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        input_var = torch.autograd.Variable(inputs)

        if args.multi_label and args.loss == 'null':
            # if multi-label and nll setting, train network by sampling an index using class weights
            if class_weights is not None:
                # normalize class weights
                target_weights = targets * torch.unsqueeze(class_weights, 0).expand_as(targets)
                sum_weights = target_weights.sum(dim=1, keepdim=True).expand_as(target_weights)
                target_weights = target_weights.div(sum_weights)
            else:
                target_weights = targets
            target_var = torch.autograd.Variable(torch.multinomial(target_weights, 1).squeeze().long())
        else:
            target_var = torch.autograd.Variable(targets)

        outputs = model(input_var)
        loss = criterion(outputs, target_var)
        loss_meter.update(loss.item(), input_var.size(0))

        # grad clearing
        optimizer.zero_grad()
        # computer grad
        loss.backward()
        # update params
        optimizer.step()

        global_step += 1

        # pbar.set_description('train loss {0}'.format(loss.item()), refresh=False)

        # --------------------------metric---------------------------------------
        if args.loss == 'nll':
            outputs = F.softmax(outputs)
        else:
            outputs = torch.sigmoid(outputs)

        if args.multi_label:
            acc, p, _, f2_score = scores(outputs.data, target_var.data, threshold)
            acc_meter.update(acc, outputs.size(0))
            precision_meter.update(p, outputs.size(0))
            f2_score_meter.update(f2_score, outputs.size(0))
        else:
            acc_1, acc_5 = accuracy(outputs.data, targets, topk=(1, 5))
            acc_top1_meter.update(acc_1[0], outputs.size(0))
            acc_top5_meter.update(acc_5[0], outputs.size(0))

        batch_time_meter.update(time.time() - start_time)

        pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                                        batch_idx * len(input_var),
                                                                        len(loader.sampler),
                                                                        100. * batch_idx / len(loader)))
        if args.multi_label:
            pbar.set_postfix_str('Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                                 'Acc {acc.val:.4f} ({acc.avg:.4f})  '
                                 'Prec {prec.val:.4f} ({prec.avg:.4f})  '
                                 'F2 {f2.val:.4f} ({f2.avg:.4f})'.format(
                batch_time=batch_time_meter, loss=loss_meter,
                acc=acc_meter, prec=precision_meter, f2=f2_score_meter),
                refresh=True)
        else:
            pbar.set_postfix_str('Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                                 'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                                 'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                batch_time=batch_time_meter, loss=loss_meter,
                top1=acc_top1_meter, top5=acc_top5_meter),
                refresh=True)

        start_time = time.time()

        # writer train log
        if (global_step + 1) % args.summary_iter == 0:
            writer.add_scalar(tag='train/loss', scalar_value=loss.cpu().item(), global_step=global_step)
            writer.add_scalar(tag='train/acc', scalar_value=acc, global_step=global_step)
            writer.add_scalar(tag='train/precision', scalar_value=p, global_step=global_step)

    if args.multi_label:
        metrics = OrderedDict([('loss', loss_meter.avg), ('acc', acc_meter.avg)])
    else:

        metrics = OrderedDict([('loss', loss_meter.avg), ('acc_top1', acc_top1_meter.avg),
                               ('acc_top5', acc_top5_meter.avg)])

    # pbar.write()
    return metrics


def eval(loader, model, epoch, criterion, threshold, use_cuda=None):
    """

    :param loader:
    :param model:
    :param epoch:
    :param criterion:
    :param threshold:
    :param use_cuda:
    :return:
    """
    model.eval()

    acc_top1_meter = AverageMeter()
    acc_top5_meter = AverageMeter()

    precision_meter = AverageMeter()
    acc_meter = AverageMeter()
    f2_score_meter = AverageMeter()

    loss_meter = AverageMeter()
    batch_time_meter = AverageMeter()


    output_list = []
    target_list = []
    start_time = time.time()
    pbar = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            input_var = torch.autograd.Variable(inputs)

            if args.multi_label and args.loss == 'nll':
                # pick one of the labels for validation loss, should we randomize like in train?
                # target_var = autograd.Variable(target.max(dim=1)[1].squeeze(), volatile=True)
                target_var = torch.autograd.Variable(targets.max(dim=1)[1].squeeze())
            else:
                # target_var = autograd.Variable(target, volatile=True)
                target_var = torch.autograd.Variable(targets)

            # calculate output
            outputs = model(input_var)
            # calculate loss

            loss = criterion(outputs, target_var)

            loss_meter.update(loss.item(), input_var.size(0))

            # --------------------------------metric-----------------------------------
            # multi label
            if args.multi_label:
                if args.loss == 'nll':
                    outputs = F.softmax(outputs)
                else:
                    outputs = torch.sigmoid(outputs)
                acc, p, _, f2_score = scores(outputs.data, target_var.data, threshold)
                acc_meter.update(acc, outputs.size(0))
                precision_meter.update(p, outputs.size(0))
                f2_score_meter.update(f2_score, outputs.size(0))
            # single label
            else:
                acc_1, acc_5 = accuracy(outputs.data, targets, topk=(1, 5))
                acc_top1_meter.update(acc_1[0], outputs.size(0))
                acc_top5_meter.update(acc_5[0], outputs.size(0))


            batch_time_meter.update(time.time() - start_time)

            pbar.set_description('Eval Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                                            batch_idx * len(input_var),
                                                                            len(loader.sampler),
                                                                            100. * batch_idx / len(loader)))

            if args.multi_label:
                pbar.set_postfix_str('Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                                     'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                                     'Acc {acc.val:.4f} ({acc.avg:.4f})  '
                                     'Prec {prec.val:.4f} ({prec.avg:.4f})  '
                                     'F2 {f2.val:.4f} ({f2.avg:.4f})'.format(
                                      batch_time=batch_time_meter, loss=loss_meter,
                                      acc=acc_meter, prec=precision_meter, f2=f2_score_meter),
                    refresh=True)
            else:
                pbar.set_postfix_str('Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                                     'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                                     'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                                     'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                                     batch_time=batch_time_meter, loss=loss_meter,
                                     top1=acc_top1_meter, top5=acc_top5_meter),
                    refresh=True)

            # record output and target to search best threshold for per category
            target_list.append(targets.cpu().numpy())
            output_list.append(outputs.data.cpu().numpy())

            start_time = time.time()
        # ----------------------------------update threshold-------------------------------------------
        output_total = np.concatenate(output_list, axis=0)
        target_total = np.concatenate(target_list, axis=0)
        if args.multi_label:
            new_threshold, f2 = optimise_f2_thresholds(target_total, output_total, verbose=False)
            metrics = [('loss', loss_meter.avg), ('acc', acc_meter.avg), ('f2', f2)]
            print('latest threshold {} => best f2-score {}'.format(new_threshold, f2))
        else:
            f2 = f2_score(output_total, target_total, threshold=0.5)
            new_threshold = []
            metrics = [('loss', loss_meter.avg), ('acc_top1', acc_top1_meter.avg), ('acc_top5', acc_top5_meter.avg),
                       ('f2', f2)]

        return OrderedDict(metrics), new_threshold


if __name__ == "__main__":
    main()


