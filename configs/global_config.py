# -*- coding: utf-8 -*-
# @Time    : 2018/9/28 19:34
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : global_config.py
# @Software: PyCharm

from easydict import EasyDict as edict
cfg = edict()

cfg.num_classes = 21



cfg.classes_weight = 'F:/projects/Deeplab v3plus/pretrained_weights/voc2012_256_class_weights'
#'/data2/linhua/VOCdevkit/pretrained_weights/voc2012_256_class_weights'
cfg.data_loader_workers = 2
cfg.pin_memory = True

cfg.batch_size = 2
cfg.epoch_num = 5
cfg.batch_save = 2
# Nesterov Momentum
cfg.lr = 0.05
cfg.momentum = 0.9
cfg.dampening = 0
cfg.nesterov = True
cfg.weight_decay = 4e-5

# Adam
cfg.betas = (0.9, 0.999)
cfg.eps = 0.1
cfg.amsgrad = 0.1

# lr_scheduler
cfg.step_size = 2
cfg.gamma = 0.94

cfg.pretrained = False