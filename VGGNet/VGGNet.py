#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/13 7:54
"""
import torch.nn as nn
from VGGNetMe import*
class VGGNet(nn.Module):
    def __init__(self, num_classes = 1000, version = 'vgg16'):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.version = version
        if self.version == 'vgg16':
            self.model = VggNet16(num_classes=self.num_classes)
        elif self.version == 'vgg19':
            self.model = VggNet19(num_classes=self.num_classes)
    def forward(self, x):
        out = self.model(x)
        return out