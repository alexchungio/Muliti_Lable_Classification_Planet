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
import numpy as np
from tqdm import tqdm

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

    # --------------------------------criterion & optimizer-----------------------
    # criterion = nn.CrossEntropyLoss()
    # optimizer = get_optimizer(model, args)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    #
    # # Resume model
    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']
    #     # for single or multi gpu
    #     if len(args.gpu_id) > 1:
    #         model.module.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #
    # # eval model
    # if args.evaluate:
    #     print('\nEvaluation only')
    #     test_loss, test_acc_1, test_acc_5 = test(val_loader, model, criterion, use_cuda)
    #     print(' Test => loss {:.4f} | acc_top1 {:.4f} acc_top5'.format(test_loss, test_acc_1, test_acc_5))
    #
    #     return None
    #
    # # best_model_weights = copy.deepcopy(model.state_dict())
    # since = time.time()
    # for epoch in range(start_epoch, args.epochs):
    #     print('Epoch {}/{} | LR {:.8f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr']))
    #
    #     train_loss, train_acc_1, train_acc_5 = train(train_loader, model, criterion, optimizer, args.summary_iter,
    #                                                  use_cuda)
    #     test_loss, test_acc_1, test_acc_5 = test(val_loader, model, criterion, use_cuda)
    #
    #     scheduler.step(metrics=test_loss)
    #
    #     # save logs
    #     writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train': train_loss, 'val': test_loss},
    #                        global_step=epoch)
    #     writer.add_scalars(main_tag='epoch/acc_top1', tag_scalar_dict={'train': train_acc_1, 'val': test_acc_1},
    #                        global_step=epoch)
    #     writer.add_scalars(main_tag='epoch/acc_top5', tag_scalar_dict={'train': train_acc_5, 'val': test_acc_5},
    #                        global_step=epoch)
    #
    #     # add learning_rate to logs
    #     writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
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

def train():
    pass

def predict():
    pass




if __name__ == "__main__":
    main()







