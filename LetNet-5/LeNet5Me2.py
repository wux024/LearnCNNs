#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/5 15:38
"""
import torch.nn as nn
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,input):
        out=input.view(input.size(0),-1)
        return out
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,6,5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            Flatten(),
            nn.Linear(400,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,10),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        logits= self.net(x)
        return logits