# -*- coding: utf-8 -*-
# @Time    : 2018/9/28 19:34
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : global_config.py
# @Software: PyCharm

from easydict import EasyDict as edict
cfg = edict()

# dataset
# cfg.dataset = "voc2012_aug"
cfg.cityscapes_path = "/data/linhua/Cityscapes"
cfg.num_classes = 21
cfg.base_size = 513
cfg.crop_size = 513
# cfg.classes_weight = True


cfg.classes_weight = '/data/linhua/VOCdevkit/pretrained_weights/'
cfg.data_loader_workers = 2
cfg.pin_memory = True

cfg.batch_size = 16
cfg.epoch_num = 46
cfg.batch_save = 50

# Nesterov Momentum
cfg.momentum = 0.9
cfg.dampening = 0
cfg.nesterov = True
cfg.weight_decay = 4e-5



# train protocol
cfg.bn_decay = 0.9997
cfg.lr_decay = 10
# cfg.iter_max = 50
cfg.poly_power = 0.9


# lr_scheduler
cfg.step_size = 30000
cfg.gamma = 0.1

