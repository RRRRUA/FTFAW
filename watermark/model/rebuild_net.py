import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from torchvision import models
import math


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResBlockRebulidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(3,128,kernel_size=3,stride=1,padding=1)
        self.rsb1 = ResnetBlock(in_channels=128, out_channels=128, conv_shortcut=False, dropout=0)
        self.rsb2 = ResnetBlock(in_channels=128, out_channels=128, conv_shortcut=False, dropout=0)
        self.rsb3 = ResnetBlock(in_channels=128, out_channels=128, conv_shortcut=False, dropout=0)
        self.rsb4 = ResnetBlock(in_channels=128, out_channels=128, conv_shortcut=False, dropout=0)
        self.conv_out = torch.nn.Conv2d(128,3,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.rsb1(x)
        x = self.rsb2(x)
        x = self.rsb3(x)
        x = self.rsb4(x)
        x = self.conv_out(x)
        return x
