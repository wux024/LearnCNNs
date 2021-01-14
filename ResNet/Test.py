#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/12 10:55
"""
import torch
from ResNet import ResNet

if __name__=='__main__':
    model = ResNet(num_classes=10, version='resnet152')
    print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)