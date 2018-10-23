# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 16:56
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : encoder.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision

import sys
sys.path.append(os.path.abspath('..'))

from graphs.models.AlignedXceptionWithoutDeformable import Xception
from graphs.models.resnet101 import ResNet101
from graphs.models.RResNet101 import resnet101


def _AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    asppconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    return asppconv

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
        self._atrous_convolution1 = _AsppConv(2048, 256, 1, 1)
        self._atrous_convolution2 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[1], dilation=atrous_rates[1])
        self._atrous_convolution3 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[2], dilation=atrous_rates[2])
        self._atrous_convolution4 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[3], dilation=atrous_rates[3])

        #image_pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.__init_weight()

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(input=input5, size=input4.size()[2:3], mode='bilinear', align_corners=True)

        return torch.cat((input1, input2, input3, input4, input5), dim=1)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, output_stride=16, pretrained=False):
        super(Encoder, self).__init__()
        #self.Xception = Xception(output_stride = 16, pretrained=pretrained)
        self.Resnet101 = resnet101(pretrained=pretrained)
        # output_stride parameter
        # self.Resnet101 = torchvision.models.resnet101(pretrained=True)
        self.ASPP = AsppModule(output_stride=output_stride)
        self.conv1 = nn.Conv2d(256*5, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.__init_weight()

    def forward(self, input):
        input, low_level_features = self.Resnet101(input)
        input = self.ASPP(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu1(input)
        # input = self.dropout(input)
        return input, low_level_features

    # def forward(self, input):
    #     input, _ = self.Resnet101(input)
    #     input = self.ASPP(input)
    #     input = self.conv1(input)
    #     input = self.bn1(input)
    #     input = self.relu1(input)
    #     input = self.dropout(input)
    #     return input

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ =="__main__":
    model = Encoder(pretrained=True)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())


