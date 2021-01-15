#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2020/12/31 16:14
"""
import torch.nn as nn
import torch.nn.functional as F
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 =   nn.Linear(16 * 5 * 5, 120)
        self.fc2 =   nn.Linear(120, 84)
        self.fc3 =   nn.Linear(84, 10)
    def forward(self, x):
        # first
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # second
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x