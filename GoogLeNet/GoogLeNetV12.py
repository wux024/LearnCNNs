#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/12 19:56
"""
import torch
import torch.nn as nn
from BasicConv2d import BasicConv2d
class Inception(nn.Module):
    def __init__(self, in_channels, out1_channels, out3in_channels, out3_channels, out5in_channels, out5_channels, outpool_channels, version = 'v1'):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.branch1 = BasicConv2d(in_channels, out1_channels, kernel_size=1)
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out3in_channels, kernel_size=1),
            BasicConv2d(out3in_channels, out3_channels, kernel_size=3, padding=1)
        )
        # 1x1 conv ->  5x5 conv branch
        # 1x1 conv ->  3x3 conv branch ->  3x3 conv branch
        if version == 'v1':
            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, out5in_channels, kernel_size=1),
                BasicConv2d(out5in_channels, out5_channels, kernel_size=3, padding=1)
            )
        elif version == 'v2':
            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, out5in_channels, kernel_size=1),
                BasicConv2d(out5in_channels, out5_channels, kernel_size=3, padding=1),
                BasicConv2d(out5_channels, out5_channels, kernel_size=3, padding=1)
            )
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, outpool_channels, kernel_size=1)
        )
    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNetV12(nn.Module):
    #input 224x224x3
    def __init__(self,num_classes=1000,version = 'v1'):
        super(GoogLeNetV12, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3,2,ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3  = BasicConv2d(64, 192, kernel_size=3, stride=1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(3,2,ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32, version)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64, version)

        self.maxpool3 = nn.MaxPool2d(3,2,ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64, version)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64, version)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64, version)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64, version)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128, version)

        self.maxpool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128, version)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128, version)

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)

        self.dropout = nn.Dropout(p=0.4)

        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
