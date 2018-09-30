# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath('..'))

from graphs.models.encoder import Encoder
from graphs.models.AlignedXceptionWithoutDeformable import SeparableConv2d


class Decoder(nn.Module):
    def __init__(self, class_num, pretrained):
        super(Decoder, self).__init__()
        self.encoder = Encoder(pretrained)
        self.conv1 = nn.Conv2d(128, 48, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)




    def forward(self, input):
        x, low_level_feature = self.encoder(input)
        low_level_feature = self.conv1(low_level_feature)
        x_4 = F.interpolate(x, scale_factor=4, mode='bilinear' ,align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)
        predict = F.interpolate(x_4_cat, scale_factor=4, mode='bilinear', align_corners=True)
        return predict

if __name__ =="__main__":
    model = Decoder(class_num=21)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output= model.forward(image)
    print(output.size())



