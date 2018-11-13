# -*- coding: utf-8 -*-
# @Time    : 2018/9/21 17:21
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : Voc_Dataset.py
# @Software: PyCharm

import PIL
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms



class Voc_Dataset(data.Dataset):
    def __init__(self,
                 root_path='/data/linhua/VOCdevkit',
                 dataset='voc2012_aug',
                 base_size=513,
                 crop_size=513,
                 is_training=True):
        """

        :param root_path:
        :param dataset:
        :param base_size:
        :param is_trainging:
        :param transforms:
        """

        self.dataset = dataset
        self.is_training = is_training
        self.base_size = base_size
        self.crop_size = crop_size

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

        elif self.dataset == 'voc2012_aug':
            self.data_path = os.path.join(root_path, "VOC2012")
            if is_training:
                item_list_filepath = os.path.join(self.data_path, "ImageSets/Segmentation/train_aug.txt")
            else:
                item_list_filepath = os.path.join(self.data_path, "ImageSets/Segmentation/val_aug.txt")

        else:
            raise Warning("dataset must be voc2007 or voc2012 or voc2012_aug")

        self.image_filepath = os.path.join(self.data_path, "JPEGImages")

        self.gt_filepath = os.path.join(self.data_path, "SegmentationClassAug")


        self.items = [id.strip() for id in open(item_list_filepath)]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']


    def __getitem__(self, item):
        id = self.items[item]

        gt_image_path = os.path.join(self.gt_filepath, "{}.png".format(id))
        gt_image = Image.open(gt_image_path)

        image_path = os.path.join(self.image_filepath, "{}.jpg".format(id))
        image = Image.open(image_path).convert("RGB")

        if self.is_training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, id

    def _train_sync_transform(self, img, mask):
        '''

        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, image):
        image_transforms = ttransforms.Compose([
            ttransforms.ToTensor(),
            ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        image = image_transforms(image)
        return image

    def _mask_transform(self, gt_image):
        target = np.array(gt_image).astype('int32')
        target = torch.from_numpy(target)

        return target




    def __len__(self):
        return len(self.items)

class VOCDataLoader():
    def __init__(self, args):

        self.args = args

        train_set = Voc_Dataset(dataset=self.args.dataset,
                                base_size=self.args.base_size,
                                crop_size=self.args.crop_size,
                                is_training=True)
        val_set = Voc_Dataset(dataset=self.args.dataset,
                              base_size=self.args.base_size,
                              crop_size=self.args.crop_size,
                              is_training=False)

        self.train_loader = data.DataLoader(train_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)

        self.train_iterations = (len(train_set) + self.args.batch_size) // self.args.batch_size
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size





if __name__ == '__main__':
    data=scipy.io.loadmat('/data/linhua/VOCdevkit/BSD/dataset/cls/2008_003846.mat')
    print(data['GTcls']["Segmentation"][0,0])
    print(np.array([[(1,2,3)]]).shape)
    print(np.array([[np.array(1), np.array(2), np.array(3)]]).shape)