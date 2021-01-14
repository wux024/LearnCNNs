#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/11 8:49
"""
import torch
import torch.nn as nn
from BasicConv2d import BasicConv2d
class Inceptionv3_modelA(nn.Module):
    def __init__(self, in_channels, out1_channels, out3in_channels, out3_channels, out5in_channels, out5_channels, outpool_channels):
        super(Inceptionv3_modelA, self).__init__()
        self.out1_channels = out1_channels
        stride = 1 if out1_channels > 0 else 2
        padding = 1 if out1_channels > 0 else 0
        if out1_channels > 0:
            self.branch1 = BasicConv2d(in_channels, out1_channels, kernel_size=1)
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out3in_channels, kernel_size=1),
            BasicConv2d(out3in_channels, out3_channels, kernel_size=3, stride=stride, padding=padding)
        )
        # 1x1 conv ->  3x3 conv branch ->  3x3 conv branch
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out5in_channels, kernel_size=1),
            BasicConv2d(out5in_channels, out5_channels, kernel_size=3, padding=1),
            BasicConv2d(out5_channels, out5_channels, kernel_size=3, stride=stride, padding=padding)
        )
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=padding),
            BasicConv2d(in_channels, outpool_channels, kernel_size=1)
        )
    def forward(self, x):
        if self.out1_channels > 0:
            y1 = self.branch1(x)
            y2 = self.branch2(x)
            y3 = self.branch3(x)
            y4 = self.branch4(x)
            out = torch.cat([y1, y2, y3, y4], 1)
        else:
            y2 = self.branch2(x)
            y3 = self.branch3(x)
            y4 = self.branch4(x)
            out = torch.cat([y2, y3, y4], 1)
        return out
class Inceptionv3_modelB(nn.Module):
    def __init__(self, in_channels, out1_channels, out3in_channels, out3_channels, out5in_channels, out5_channels, outpool_channels):
        super(Inceptionv3_modelB, self).__init__()
        self.out1_channels = out1_channels
        stride = 1 if out1_channels > 0 else 2
        padding = 1 if out1_channels > 0 else 0
        if out1_channels > 0:
            self.branch1 = BasicConv2d(in_channels, out1_channels, kernel_size=1)
            # 1x1 conv -> 1x7 conv branch -> 7x1 conv branch
            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, out3in_channels, kernel_size=1),
                BasicConv2d(out3in_channels, out3_channels, kernel_size=(1, 7), padding = (0,3)),
                BasicConv2d(out3_channels, out3_channels, kernel_size=(7, 1), padding = (3,0))
            )
            # 1x1 conv ->  1x7 conv branch ->  7x1 conv branch -> 1x7 conv branch ->  7x1 conv branch
            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, out5in_channels, kernel_size=1),
                BasicConv2d(out5in_channels, out5_channels, kernel_size=(1, 7), padding = (0,3)),
                BasicConv2d(out5_channels, out5_channels, kernel_size=(7, 1), padding = (3,0)),
                BasicConv2d(out5_channels, out5_channels, kernel_size=(1, 7), padding = (0,3)),
                BasicConv2d(out5_channels, out5_channels, kernel_size=(7, 1), padding = (3,0)),
            )
        else:
            # 1x1 conv -> 1x7 conv branch -> 7x1 conv branch
            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, out3in_channels, kernel_size=1),
                BasicConv2d(out3in_channels, out3_channels, kernel_size=3, stride=stride, padding=padding),
            )
            # 1x1 conv ->  1x7 conv branch ->  7x1 conv branch -> 1x7 conv branch ->  7x1 conv branch
            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, out5in_channels, kernel_size=1),
                BasicConv2d(out5in_channels, out5_channels, kernel_size=(1, 7), padding=(0, 3)),
                BasicConv2d(out5_channels, out5_channels, kernel_size=(7, 1), padding=(3, 0)),
                BasicConv2d(out5_channels, out5_channels, kernel_size=3, stride=stride, padding=padding),
            )
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=padding),
            BasicConv2d(in_channels, outpool_channels, kernel_size=1)
        )
    def forward(self, x):
        if self.out1_channels > 0:
            y1 = self.branch1(x)
            y2 = self.branch2(x)
            y3 = self.branch3(x)
            y4 = self.branch4(x)
            out = torch.cat([y1, y2, y3, y4], 1)
        else:
            y2 = self.branch2(x)
            y3 = self.branch3(x)
            y4 = self.branch4(x)
            out = torch.cat([y2, y3, y4], 1)
        return out

class Inceptionv3_modelC(nn.Module):
    def __init__(self, in_channels, out1_channels, out3in_channels, out3_channels, out5in_channels, out5_channels, outpool_channels):
        super(Inceptionv3_modelC, self).__init__()
        self.branch1 = BasicConv2d(in_channels, out1_channels, kernel_size=1)

        self.branch2_0 = BasicConv2d(in_channels, out3in_channels, kernel_size=1)
        self.branch2_1 = BasicConv2d(out3in_channels, out3_channels, kernel_size=(3, 1), padding = (1,0))
        self.branch2_2 = BasicConv2d(out3in_channels, out3_channels, kernel_size=(1, 3), padding = (0,1))

        self.branch3_0 = BasicConv2d(in_channels, out5in_channels, kernel_size=1)
        self.branch3_1 = nn.Sequential(
            BasicConv2d(out5in_channels, out5_channels, kernel_size=(3, 3), padding=1),
            BasicConv2d(out5_channels, out5_channels, kernel_size=(1, 3), padding = (0,1)),
        )
        self.branch3_2 = nn.Sequential(
            BasicConv2d(out5in_channels, out5_channels, kernel_size=(3, 3), padding=1),
            BasicConv2d(out5_channels, out5_channels, kernel_size=(3, 1), padding = (1,0)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, outpool_channels, kernel_size=1)
        )
    def forward(self, x):
        y1 = self.branch1(x)
        y20 = self.branch2_0(x)
        y2 = self.branch2_1(y20)
        y3 = self.branch2_2(y20)
        y30 = self.branch3_0(x)
        y4 = self.branch3_1(y30)
        y5 = self.branch3_2(y30)
        y6 = self.branch4(x)
        out = torch.cat([y1, \
                          torch.cat([y2, y3], 1), \
                          torch.cat([y4, y5], 1), \
                        y6], 1)
        return out
class GoogLeNetV3(nn.Module):
    def __init__(self, num_classes = 1000, version = 'v3'):
        # 299 x 299 x 3
        super(GoogLeNetV3, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride = 2),
            BasicConv2d(32, 32, kernel_size=3, stride = 1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding = 1),
            nn.MaxPool2d(3,2),
            BasicConv2d(64, 80, kernel_size=3, stride=1),
            BasicConv2d(80, 192, kernel_size=3, stride=2),
            BasicConv2d(192, 288, kernel_size=3, stride=1, padding = 1)
        )

        self.inception3a = Inceptionv3_modelA(288, 64, 64, 96, 48, 64, 64)
        self.inception3b = Inceptionv3_modelA(288, 64, 64, 96, 48, 64, 64)
        self.inception3c = Inceptionv3_modelA(288, 0, 192, 384, 64, 96, 288)

        self.inception5a = Inceptionv3_modelB(768, 192, 160, 192, 160, 192, 192)
        self.inception5b = Inceptionv3_modelB(768, 192, 160, 192, 160, 192, 192)
        self.inception5c = Inceptionv3_modelB(768, 192, 160, 192, 160, 192, 192)
        self.inception5d = Inceptionv3_modelB(768, 192, 160, 192, 160, 192, 192)
        self.inception5e = Inceptionv3_modelB(768, 0, 192, 320, 192, 192, 768)

        self.inception2a = Inceptionv3_modelC(1280, 320, 384, 384, 448, 384, 192)
        self.inception2b = Inceptionv3_modelC(2048, 320, 384, 384, 448, 384, 192)

        self.pool = nn.MaxPool2d(8,1)

        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception5c(x)
        x = self.inception5d(x)
        x = self.inception5e(x)

        x = self.inception2a(x)
        x = self.inception2b(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
