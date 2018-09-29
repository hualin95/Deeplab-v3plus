# -*- coding: utf-8 -*-
# @Time    : 2018/9/27 17:06
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : data_utils.py
# @Software: PyCharm

import numpy as np
from datasets.Voc_Dataset import VOCDataLoader
def calculate_weigths_labels(config):
    # Create an instance from the data loader
    from tqdm import tqdm
    data_loader = VOCDataLoader(config)
    z = np.zeros((config.num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(data_loader.train_loader, total=data_loader.train_iterations)

    for _, y in tqdm_batch:
        labels = y.numpy().astype(np.uint8).ravel().tolist()
        z += np.bincount(labels, minlength=config.num_classes)
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
    np.save('/data2/linhua/VOCdevkit/pretrained_weights/voc2012_256_class_weights', ret)
    print(ret)