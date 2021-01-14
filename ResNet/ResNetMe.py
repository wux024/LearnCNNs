#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/12 12:40
"""
import torch.nn as nn
def BRMConv2d(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, same_shape=True):
        super(ResidualBlock, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.block(x)
        if not self.same_shape:
            residual = self.bn3(self.conv3(x))
        out += residual
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, same_shape=True, bottle=True, expansion = 4):
        super(Bottleneck, self).__init__()
        self.same_shape = same_shape
        self.bottle = bottle
        self.expansion = expansion
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels*self.expansion)
        )
        if bottle:
            self.conv4 = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=strides, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.bottle:
            residual = self.bn4(self.conv4(x))
        out += residual
        out = self.relu(out)
        return out

class ResNetMe(nn.Module):
    def __init__(self, blocks, version = 0, num_classes=1000, expansion = 4):
        super(ResNetMe,self).__init__()
        self.expansion = expansion
        self.conv1 = BRMConv2d(3, 64)
        if version == 0:
            self.layer1 = self.make_layer0(64, 64,  blocks[0])
            self.layer2 = self.make_layer0(64, 128,  blocks[1], stride=2)
            self.layer3 = self.make_layer0(128, 256, blocks[2], stride=2)
            self.layer4 = self.make_layer0(256, 512, blocks[3], stride=2)
            self.fc = nn.Linear(512, num_classes)
        else:
            self.layer1 = self.make_layer1(64, 64, blocks[0], stride=1)
            self.layer2 = self.make_layer1(256, 128, blocks[1], stride=2)
            self.layer3 = self.make_layer1(512, 256, blocks[2], stride=2)
            self.layer4 = self.make_layer1(1024, 512, blocks[3], stride=2)
            self.fc = nn.Linear(2048, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
    def make_layer0(self, in_channels, out_channels, block_num, stride=1):
        layers = []
        if stride != 1:
            layers.append(ResidualBlock(in_channels, out_channels, stride, same_shape=False))
        else:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer1(self, in_channels, out_channels, block, stride):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels,stride, bottle=True))
        for i in range(1, block):
            layers.append(Bottleneck(out_channels*self.expansion, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
