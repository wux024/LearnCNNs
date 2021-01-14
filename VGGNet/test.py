#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/9 13:20
"""
import torch
from VGGNet import VGGNet
model = VGGNet(num_classes=10, version='vgg16')
input = torch.randn(20,3,224,224)
output = model(input)
print(model)
print(output.shape)