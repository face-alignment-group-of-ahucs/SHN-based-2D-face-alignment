#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:44:00 2018

@author: xiang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        planes = int(out_planes/2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsample,self).__init__()
        self.upsample = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.upsample(x)

class HourGlass(nn.Module):
    def __init__(self, depth, num_features):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = num_features
        self.Upsample = Upsample(256,256)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = self.Upsample(low3)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class FAN(nn.Module):

    def __init__(self, inplanes, outplanes, bn=False):
        super(FAN, self).__init__()
        self.bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(inplanes)

        # Base part
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = HourGlass(4, 256)
        self.conv5 = ConvBlock(256,128)
        self.conv6 = conv3x3(128, outplanes)
        self.Upsample = Upsample(128,128)

    def forward(self, x):
        
        if self.bn:
            x = self.bn(x)
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2,stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.Upsample(x)
        out = self.conv6(x)

        return out
        