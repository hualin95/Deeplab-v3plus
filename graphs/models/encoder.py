# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 16:56
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : encoder.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
# from Xception import Xception
import sys
sys.path.append('/home/linhua/projects/deeplab/')
from graphs.models.AlignedXceptionWithoutDeformable import Xception

class AsppModule(nn.Module):
    def __init__(self, output_stride=16):
        super(AsppModule, self).__init__()

        # output_stride choice
        if output_stride ==16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2*[0, 6, 12, 18]
        else:
            raise Warning("atrous_rates must be 8 or 16!")
        # atrous_spatial_pyramid_pooling part
        self._atrous_convolution1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self._atrous_convolution2 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1]),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self._atrous_convolution3 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2]),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self._atrous_convolution4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=atrous_rates[3], dilation=atrous_rates[3]),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        #image_pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(input=input5, size=input4.size()[2:3], mode='bilinear', align_corners=True)

        return torch.cat((input1, input2, input3, input4, input5), dim=1)


class Encoder(nn.Module):
    def __init__(self, pretrained):
        super(Encoder, self).__init__()
        self.Xception = Xception(output_stride = 16, pretrained=pretrained)
        # output_stride parameter
        self.ASPP = AsppModule(output_stride=16)
        self.conv1 = nn.Conv2d(256*5, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

    def forward(self, input):
        input, low_level_features = self.Xception(input)
        input = self.ASPP(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu1(input)
        return input, low_level_features

if __name__ =="__main__":
    model = Encoder()
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output, low_l_f = model.forward(image)
    print(output.size())


