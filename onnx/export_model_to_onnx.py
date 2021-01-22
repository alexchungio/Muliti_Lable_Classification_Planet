#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : export_model_to_onnx.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/22 下午1:38
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import onnx
import onnxruntime  # onnxruntime-gpu==1.1.0 <=> cuda==10.0  # onnxruntime-gpu>=1.2.0 <==> cuda==10.1

from configs.cfgs import args
from models.model_factory import build_model
from utils.tools import AverageMeter
from utils.misc import read_class_names, index_to_tag

batch_size = 1
onnx_model_path = '../outputs/weights/best_model.onnx'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def convert_onnx(torch_model, batch_size, onnx_model_path):
    # set the model to inference mode
    torch_model.eval()
    input_names = ['input']
    output_names = ['output']
    input = torch.randn(batch_size, 3, args.image_size, args.image_size, requires_grad=True, device=device)
    torch.onnx.export(torch_model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      onnx_model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,  # the model's input names
                      output_names=output_names,  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # dynamic batch_size
                                    'output':{0: 'batch_size'}},
                      # dynamic_axes = {'input': [2, 3],  # dynamic input shape
                      #                 'output':[2, 3]},
                      verbose=True) # show verbose
    print('Successful convert torch model to onnx model')

def check_onnx(onnx_model_path):
    """

    :param onnx_model_path:
    :return:
    """
    onnx_model = onnx.load_model(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    print('Passing check')


def convert_to_numpy(tensor):

    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def compare_torch_and_onnx_model(torch_model, onnx_model_path):
    """

    :param torch_model:
    :param onnx_model:
    :return:
    """
    x = torch.randn(batch_size, 3, args.image_size, args.image_size, requires_grad=True)

    # Run the model on the backend
    ort_session = onnxruntime.InferenceSession(onnx_model_path, None)

    # get the name of the first input of the model
    input = ort_session.get_inputs()[0]
    print('Input Name {} shape {}:', input.name, input.shape)
    output = ort_session.get_outputs()[0]
    print('Input Name {} shape {}:', output.name, output.shape)

    # compute ONNX Runtime output prediction
    start_time = time.perf_counter()
    ort_inputs = {ort_session.get_inputs()[0].name: convert_to_numpy(x)}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_cost_time = time.perf_counter() - start_time

    # compute torch model prediction
    start_time = time.perf_counter()
    torch_outputs = torch_model(x.to(device))
    torch_cost_time = time.perf_counter() - start_time

    print('pytorch cost time {}'.format(torch_cost_time))
    print('onnx_cost_time {}'.format(onnx_cost_time))

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(convert_to_numpy(torch_outputs), ort_outputs[0], rtol=1e-03, atol=1e-05)

    print('Compare Done')




def main():
    torch_model = build_model(model_name=args.model_name, num_classes=args.num_classes, global_pool=args.global_pool)

    if args.best_checkpoint is not None:
        assert os.path.isfile(args.best_checkpoint), '{} not found'.format(args.best_checkpoint)
        checkpoint = torch.load(args.best_checkpoint, map_location=None)
        # Load all tensors onto the CPU
        # checkpoint = torch.load(args.best_checkpoint, map_location=lambda storage, loc: storage)
        # >>> torch.load('tensors.pt')
        # # Load all tensors onto the CPU
        # >>> torch.load('tensors.pt', map_location=torch.device('cpu'))
        # # Load all tensors onto the CPU, using a function
        # >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
        # # Load all tensors onto GPU 1
        # >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
        # # Map tensors from GPU 1 to GPU 0
        # >>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})

        print('Restoring model with {} architecture...'.format(checkpoint['arch']))
        # load model weights
        torch_model.load_state_dict(checkpoint['state_dict'])


        if checkpoint['num_gpu'] > 1:
            torch_model = torch.nn.DataParallel(torch_model).cuda()
        else:
            torch_model.cuda()

    # convert to onnx
    convert_onnx(torch_model, batch_size=batch_size, onnx_model_path=onnx_model_path)


    # verify the model’s structure and confirm that the model has a valid schema
    check_onnx(onnx_model_path)

    compare_torch_and_onnx_model(torch_model, onnx_model_path)

if __name__ == "__main__":
    main()

# onnx_cost_time 0.10371108900289983
# pytorch cost time 0.030497150000883266
