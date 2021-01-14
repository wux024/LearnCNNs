#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/14 21:06
"""
import torch
from LeNet5Me import LeNet5
model = LeNet5()
input = torch.randn(10,1,28,28)
output = model(input)
print(model)
print(output.shape)