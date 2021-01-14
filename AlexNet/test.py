#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/14 20:50
"""
import torch
from AlexNetMe import AlexNet

model = AlexNet(num_classes=150)
print(model)
input = torch.randn(8,3,227,227)
output = model(input)
print(output.shape)