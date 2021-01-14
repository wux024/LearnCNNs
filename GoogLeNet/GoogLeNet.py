#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/12 19:47
"""
import torch.nn as nn
from GoogLeNetV12 import GoogLeNetV12
from GoogLeNetV3 import GoogLeNetV3
from GoogLeNetV4 import GoogLeNetV4
class GoogLeNet(nn.Module):
    def __init__(self, num_classes = 1000, version = 'v1'):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.version = version
        if self.version == 'v1':
            self.model = GoogLeNetV12(num_classes=self.num_classes, version=self.version)
        elif self.version == 'v2':
            self.model = GoogLeNetV12(num_classes=self.num_classes, version=self.version)
        elif self.version == 'v3':
            self.model= GoogLeNetV3(num_classes=self.num_classes, version=self.version)
        elif self.version == 'v4':
            self.model= GoogLeNetV4(num_classes=self.num_classes, version=self.version)
        elif self.version == 'res-v1':
            self.model = GoogLeNetV4(num_classes=self.num_classes, version=self.version)
        elif self.version == 'res-v2':
            self.model = GoogLeNetV4(num_classes=self.num_classes, version=self.version)
    def forward(self, x):
        out = self.model(x)
        return out