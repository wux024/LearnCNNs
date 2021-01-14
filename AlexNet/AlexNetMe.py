#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/5 19:13
"""
import torch.nn as nn
import torch.nn.functional as F
class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        self.pool = nn.MaxPool2d(3, 2)
        self.LRN  = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
        self.conv1 = nn.Conv2d(3, 96, 11, 4, 0)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        # 1st
        x = self.LRN(self.pool(F.relu(self.conv1(x))))
        # 2ed
        x = self.LRN(self.pool(F.relu(self.conv2(x))))
        # 3rd
        x = F.relu(self.conv3(x))
        # 4th
        x = F.relu(self.conv4(x))
        # 5th
        x = self.pool(F.relu(self.conv5(x)))
        # fc1
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.drop(x)))
        # fc2
        x = F.relu(self.fc2(self.drop(x)))
        # fc2
        x = self.fc3(x)
        return x