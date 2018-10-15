# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 10:41
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : AlignedXceptionWithoutDeformable.py
# @Software: PyCharm


import math
import logging
from torchsummary import summary
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()

        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        padding = pad_total // 2

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        # extra BatchNomalization and ReLU
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters, out_filters, reps=3, strides=1, start_with_relu=True, grow_first=True, dilation=1):
        '''

        :param in_filters:
        :param out_filters:
        :param reps:
        :param strides:
        :param start_with_relu:
        :param grow_first: whether add channels at first
        '''
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))

            rep.append(self.relu)
            rep.append(SeparableConv2d(out_filters, out_filters, 3, stride=1, bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, in_filters, 3, stride=1, bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(in_filters))

            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_filters, out_filters, 3, stride=strides, bias=False, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, output_stride, pretrained=True):
        super(Xception, self).__init__()

        if output_stride ==8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            # atrous_rates = (12, 24, 36)
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            # atrous_rates = (6, 12, 18)
        else:
            raise Warning("atrous_rates must be 8 or 16!")

        self.conv1 = nn.Conv2d(3, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 3, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 3, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 3, strides=entry_block3_stride, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)

        self.block12 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block13 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block14 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block15 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)

        self.block16 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block17 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block18 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)
        self.block19 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, dilation=middle_block_rate)

        self.block20 = Block(728, 1024, 3, 1, start_with_relu=True, grow_first=False, dilation=exit_block_rates[0])

        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, dilation=exit_block_rates[1])
        self.bn3 = nn.BatchNorm2d(1536)
        # do relu here

        self.conv4 = SeparableConv2d(1536, 1536, kernel_size=3, stride=1, dilation=exit_block_rates[1])
        self.bn4 = nn.BatchNorm2d(1536)
        # do relu here

        self.conv5 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, dilation=exit_block_rates[1])
        self.bn5 = nn.BatchNorm2d(2048)
        # do relu here

        self._init_weights()
        if pretrained is not False:
            self._load_xception_weight()


    def forward(self, input):
        # Entry flow
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Middle flow
        x = self.block1(x)
        low_level_features = x
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        #Exit flow
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_features

    def _load_xception_weight(self):
        print("Loading pretrained weights in Imagenet...")
        pretrained_dict = model_zoo.load_url(url="http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth",
                                             model_dir="/data/linhua/VOCdevkit/")
        model_dict = self.state_dict()
        new_dict = {}

        for k, v in pretrained_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    new_dict[k] = v
                    new_dict[k.replace('block11', 'block12')] = v
                    new_dict[k.replace('block11', 'block13')] = v
                    new_dict[k.replace('block11', 'block14')] = v
                    new_dict[k.replace('block11', 'block15')] = v
                    new_dict[k.replace('block11', 'block16')] = v
                    new_dict[k.replace('block11', 'block17')] = v
                    new_dict[k.replace('block11', 'block18')] = v
                    new_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    new_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    new_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    new_dict[k] =v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

        #------- init weights --------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------
if __name__ == '__main__':
    model = Xception(output_stride=16, pretrained=False)
    # print(model.state_dict)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    # with torch.no_grad():
    #    output, low_l_f = model.forward(image)
    # print(low_l_f.size())
    # print(model)
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =model.to(device)
    summary(model,(3, 512, 512))
