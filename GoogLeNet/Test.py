#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/10 17:15
"""
import torch
from GoogLeNet import GoogLeNet
# v1
model1 = GoogLeNet(num_classes = 10, version='v1')
model2 = GoogLeNet(num_classes = 10, version='v2')
input = torch.randn(1, 3, 224, 224)
output1 = model1(input)
output2 = model2(input)
print(output1.shape)
print(output2.shape)
# v3
model3 = GoogLeNet(num_classes=10, version='v3')
input = torch.randn(1, 3, 299, 299)
output3 = model3(input)
print(output3.shape)
# v4
model4 = GoogLeNet(num_classes=10, version='v4')
input = torch.randn(1, 3, 299, 299)
output4 = model4(input)
print(output4.shape)
# v4-res-v1
model5 = GoogLeNet(num_classes=10, version='res-v1')
input = torch.randn(1, 3, 299, 299)
output5 = model5(input)
print(output5.shape)
# v4-res-v2
model6 = GoogLeNet(num_classes=10, version='res-v2')
input = torch.randn(1, 3, 299, 299)
output6= model6(input)
print(output6.shape)