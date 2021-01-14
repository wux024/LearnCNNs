#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/1/5 20:11
"""
import torch.nn as nn
class AlexNet(nn.Module):
  def __init__(self,num_classes=1000):
    super(AlexNet,self).__init__()
    self.feature_extraction = nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0,bias=False),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
      nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2,bias=False),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
      nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(in_features=256*6*6,out_features=4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=num_classes),
    )
  def forward(self,x):
    x = self.feature_extraction(x)
    x = x.view(x.size(0),-1)
    x = self.classifier(x)
    return x

