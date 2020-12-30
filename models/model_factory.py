#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model_factory.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/11 下午2:58
# @ Software   : PyCharm
#-------------------------------------------------------
from torch import nn
import models as model_zoo

model_names = sorted(name for name in model_zoo.__dict__
                                 if not name.startswith("__")
                                 and callable(model_zoo.__dict__[name]))

def build_model(model_name='resnet50', pretrained=True, num_classes=1000, **kwargs):

    global_pool = kwargs.get('global_pool', 'avg')
    if model_name == 'resnet18':
        if pretrained:
            model = model_zoo.resnet18(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.resnet18(num_classes=num_classes, global_pool=global_pool, **kwargs)
    if model_name == 'resnet34':
        if pretrained:
            model = model_zoo.resnet34(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.resnet34(num_classes=num_classes, global_pool=global_pool, **kwargs)
    if model_name == 'resnet50':
        if pretrained:
            model = model_zoo.resnet50(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.resnet50(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'resnet101':
        if pretrained:
            model = model_zoo.resnet101(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.resnet101(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'resnet152':
        if pretrained:
            model = model_zoo.resnet152(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.resnet152(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet121':
        if pretrained:
            model = model_zoo.densenet121(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.densenet121(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet161':
        if pretrained:
            model = model_zoo.densenet161(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.densenet161(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet169':
        if pretrained:
            model = model_zoo.densenet169(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.densenet169(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet201':
        if pretrained:
            model = model_zoo.densenet201(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = model_zoo.densenet201(num_classes=num_classes, global_pool=global_pool, **kwargs)
    else:
        assert False and "Invalid model"

    return model


def main():
    resnet_50 = build_model(num_classes=17)
    print(model_names)

if __name__ == "__main__":
    main()
