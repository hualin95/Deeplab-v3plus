# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 17:21
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : Voc_Dataset.py
# @Software: PyCharm

import PIL
import scipy.io
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms




class Voc_Dataset(data.Dataset):
    def __init__(self, root_path='/data/linhua/VOCdevkit', dataset='voc2012', image_size=512, is_training=True,
                 image_transforms=None, gt_image_transforms=None):
        """

        :param root_path:
        :param dataset:
        :param image_size:
        :param is_trainging:
        :param transforms:
        """
        self.dataset = dataset
        self.is_training = is_training
        if self.dataset == 'voc2007':
            self.data_path = os.path.join(root_path, "VOC2007")
            if is_training:
                item_list_filepath = os.path.join(self.data_path, "ImageSets/Segmentation/trainval.txt")
            else:
                item_list_filepath = os.path.join(self.data_path, "ImageSets/Segmentation/test.txt")

        elif self.dataset == 'voc2012':
            self.data_path = os.path.join(root_path, "VOC2012")
            if is_training:
                item_list_filepath = os.path.join(self.data_path, "ImageSets/Segmentation/train.txt")
            else:
                item_list_filepath = os.path.join(self.data_path, "ImageSets/Segmentation/val.txt")

        elif self.dataset == 'bsd':
            self.data_path = os.path.join(root_path, "BSD")
            if is_training:
                item_list_filepath = os.path.join(self.data_path, "dataset/train.txt")
            else:
                item_list_filepath = os.path.join(self.data_path, "dataset/val.txt")

        else:
            raise Warning("dataset must be voc2007 or voc2012 or bsd")

        if self.dataset == 'voc2007' or self.dataset == 'voc2012':
            self.image_filepath = os.path.join(self.data_path, "JPEGImages")
            self.gt_filepath = os.path.join(self.data_path, "SegmentationClass")

        elif self.dataset == 'bsd':
            self.image_filepath = os.path.join(self.data_path, "dataset/img")
            self.gt_filepath = os.path.join(self.data_path, "dataset/cls_png")

        else:
            raise Warning("dataset must be voc2007 or voc2012 or bsd")

        self.items = [id.strip() for id in open(item_list_filepath)]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.is_training = is_training
        self.im_transforms = image_transforms
        self.gt_transforms = gt_image_transforms

    def __getitem__(self, item):
        id = self.items[item]
        if self.dataset == 'bsd':
            gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            gt_image = Image.open(gt_image_path)
        else:
            gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
            gt_image = Image.open(gt_image_path).convert('P')

        image_path = os.path.join(self.image_filepath, "{}.jpg".format(id))
        image = Image.open(image_path).convert("RGB")

        if transforms:
            image = self.im_transforms(image)
            gt_image =self.gt_transforms(gt_image)
        return image, gt_image, id



    def __len__(self):
        return len(self.items)

class VOCDataLoader():
    def __init__(self, config):
        mean_std = ([122.675, 116.669, 104.008], [1.0, 1.0, 1.0])

        self.config = config
        self.transform_image = transforms.Compose([
            transforms.Resize((512, 512), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul_(255)),
            transforms.Normalize(*mean_std)
        ])

        self.transform_gt = transforms.Compose([
            transforms.Resize((512, 512), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor()
        ])

        train_set = Voc_Dataset(dataset='bsd', image_transforms=self.transform_image, gt_image_transforms=self.transform_gt)
        val_set = Voc_Dataset(dataset='voc2012', is_training=False, image_transforms=self.transform_image,
                              gt_image_transforms=self.transform_gt)

        self.train_loader = data.DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)
        self.valid_loader = data.DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)
        self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
        self.valid_iterations = (len(val_set) + self.config.batch_size) // self.config.batch_size


if __name__ == '__main__':
    data=scipy.io.loadmat('/data/linhua/VOCdevkit/BSD/dataset/cls/2008_003846.mat')
    print(data['GTcls']["Segmentation"][0,0])
    print(np.array([[(1,2,3)]]).shape)
    print(np.array([[np.array(1), np.array(2), np.array(3)]]).shape)