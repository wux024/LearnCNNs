#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/13 7:44
"""
import torch.nn as nn
from ResNetMe import ResNetMe
class ResNet(nn.Module):
    def __init__(self, num_classes = 1000, version = 'resnet50'):
        super(ResNet, self).__init__()
        self.version = version
        self.num_classes = num_classes
        if self.version == 'resnet18':
            self.model = ResNetMe([2, 2, 2, 2],num_classes=self.num_classes)
        elif self.version == 'resnet34':
            self.model = ResNetMe([3, 4, 6, 3],num_classes=self.num_classes)
        elif self.version == 'resnet50':
            self.model = ResNetMe([3, 4, 6, 3],version = 1,num_classes=self.num_classes)
        elif self.version == 'resnet101':
            self.model = ResNetMe([3, 4, 23, 3],version = 1,num_classes=self.num_classes)
        elif self.version == 'resnet152':
            self.model = ResNetMe([3, 8, 36, 3],version = 1,num_classes=self.num_classes)
    def forward(self, x):
        out = self.model(x)
        return out
