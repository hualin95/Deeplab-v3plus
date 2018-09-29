# -*- coding: utf-8 -*-
# @Time    : 2018/9/17 9:48
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : deeplab.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.functional as F


class DEEPLABV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
