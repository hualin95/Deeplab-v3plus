# -*- coding: utf-8 -*-
# @Time    : 2018/9/27 17:06
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : data_utils.py
# @Software: PyCharm

import os
import sys
import time
import torch
sys.path.append(os.path.abspath('..'))
from tqdm import tqdm
import numpy as np
from datasets.Voc_Dataset import VOCDataLoader
from configs.global_config import cfg

def calculate_weigths_labels(config):
    # Create an instance from the data loader
    data_loader = VOCDataLoader(config)
    z = np.zeros((config.num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(data_loader.train_loader, total=data_loader.train_iterations)

    for _, y, _ in tqdm_batch:
        y = y.numpy()*255
        mask = (y >= 0) & (y < config.num_classes)
        labels = y[mask].astype(np.uint8) #.ravel().tolist()
        count_l = np.bincount(labels, minlength=config.num_classes)
        z += count_l
    tqdm_batch.close()
    # ret = compute_class_weight(class_weight='balanced', classes=np.arange(21), y=np.asarray(labels, dtype=np.uint8))
    total_frequency = np.sum(z)
    print(z)
    print(total_frequency)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join('/data/linhua/VOCdevkit/pretrained_weights',config.dataset+'classes_weights')
    np.save(classes_weights_path, ret)
    print(ret)

if __name__ =='__main__':
    config = cfg
    calculate_weigths_labels(config)