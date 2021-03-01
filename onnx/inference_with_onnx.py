#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference_with_onnx.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/22 下午1:39
# @ Software   : PyCharm
#-------------------------------------------------------

import onnxruntime
import torch
from PIL import Image
from data.transform import get_transform
import numpy as np


class ONNXModel(object):

    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
        super(ONNXModel,self).__init__()

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = self.convert_to_numpy(image_tensor)
        return input_feed

    def detect(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''

        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)

        return outputs

    def convert_to_numpy(self, tensor):

        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_image(path):
    """

    :param path:
    :return:
    """
    image = Image.open(path)
    return image.convert('RGB')


def get_img_tensor(path, transform):
    """

    :param path:
    :return:
    """
    rgb_img = load_image(path)
    img_tensor = transform(rgb_img)

    return img_tensor


def main():

    onnx_model_path = '../outputs/weights/best_model.onnx'
    demo_path = '../docs/demo/demo.jpg'
    threshold = [0.05, 0.08, 0.14, 0.13, 0.21, 0.16, 0.23, 0.1, 0.26, 0.17, 0.25, 0.17, 0.21, 0.21, 0.15, 0.18, 0.21]
    threshold = torch.tensor(threshold, dtype=torch.float32)

    ort = ONNXModel(onnx_model_path)
    test_transform = get_transform(size=256, mode='test')
    img_tensor = get_img_tensor(demo_path, test_transform)
    img_batch = img_tensor.unsqueeze(dim=0)

    outputs = torch.tensor(ort.detect(img_batch)[0], dtype=torch.float32)
    predict = torch.sigmoid(outputs)

    expand_threshold = torch.unsqueeze(threshold, 0).expand_as(predict)
    output_labels = (predict.data > expand_threshold).byte()

    # move data to CPU and collect
    predict = predict.cpu().data.numpy()
    output_labels = output_labels.cpu().numpy()

    print(predict)
    print(output_labels)


if __name__ == "__main__":
    main()

