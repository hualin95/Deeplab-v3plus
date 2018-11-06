# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 15:20
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : cityscapes_Dataset.py
# @Software: PyCharm


import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms



class City_Dataset(data.Dataset):
    def __init__(self,
                 data_path='/data/linhua/Cityscapes',
                 list_path='/city_list',
                 split='train',
                 base_size=520,
                 crop_size=480):
        """

        :param root_path:
        :param dataset:
        :param base_size:
        :param is_trainging:
        :param transforms:
        """

        self.data_path = data_path
        self.list_path = list_path
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size

        if self.split == 'train':
            item_list_filepath = os.path.join(self.list_path, "train.txt")

        elif self.split == 'val':
            item_list_filepath = os.path.join(self.list_path, "val.txt")

        elif self.split == 'trainval':
            item_list_filepath = os.path.join(self.list_path, "trainval.txt")

        elif self.split == 'test':
            item_list_filepath = os.path.join(self.list_path, "test.txt")

        else:
            raise Warning("split must be train/val/trainavl/test")

        self.image_filepath = os.path.join(self.data_path, "leftImg8bit_trainvaltest/leftImg8bit")

        self.gt_filepath = os.path.join(self.data_path, "gtFine_trainvaltest/gtFine")


        self.items = [id.strip() for id in open(item_list_filepath)]



    def __getitem__(self, item):
        id = self.items[item]
        filename = id.split("train_")[-1].split("val_")[-1]

        self.image_filepath = os.path.join(self.image_filepath, id.split("_")[0], id.split("_")[1])
        image_filename = filename + "_leftImg8bit.png"
        image_path = os.path.join(self.image_filepath, image_filename)
        image = Image.open(image_path).convert("RGB")

        if self.split == "test":
            return self._test_transform(image), filename

        self.gt_filepath = os.path.join(self.gt_filepath, id.split("_")[0], id.split("_")[1])
        gt_filename = filename + "_gtFine_labelIds.png"
        gt_image_path = os.path.join(self.gt_filepath, gt_filename)
        gt_image = Image.open(gt_image_path)

        if self.split == "train" or self.split == "trainval":
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, filename



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

    def _test_transform(self, img):
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
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img = self._img_transform(img)
        return img

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

class City_DataLoader():
    def __init__(self, split, config):

        self.config = config

        # train_set = City_Dataset(data_path=self.config.cityscapes_path,
        #                         list_path='/city_list',
        #                         split='train',
        #                         base_size=self.config.base_size,
        #                         crop_size=self.config.crop_size)
        #
        # val_set = City_Dataset(data_path=self.config.cityscapes_path,
        #                         list_path='/city_list',
        #                         split='val',
        #                         base_size=self.config.base_size,
        #                         crop_size=self.config.crop_size)
        #
        # trainval_set = City_Dataset(data_path=self.config.cityscapes_path,
        #                         list_path='/city_list',
        #                         split='trainval',
        #                         base_size=self.config.base_size,
        #                         crop_size=self.config.crop_size)
        #
        # test_set = City_Dataset(data_path=self.config.cityscapes_path,
        #                         list_path='/city_list',
        #                         split='test',
        #                         base_size=self.config.base_size,
        #                         crop_size=self.config.crop_size)
        #
        # self.train_loader = data.DataLoader(train_set,
        #                                     batch_size=self.config.batch_size,
        #                                     shuffle=True,
        #                                     num_workers=self.config.data_loader_workers,
        #                                     pin_memory=self.config.pin_memory,
        #                                     drop_last=True)
        # self.valid_loader = data.DataLoader(val_set,
        #                                     batch_size=self.config.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.config.data_loader_workers,
        #                                     pin_memory=self.config.pin_memory,
        #                                     drop_last=True)
        # self.trainval_loader = data.DataLoader(trainval_set,
        #                                     batch_size=self.config.batch_size,
        #                                     shuffle=True,
        #                                     num_workers=self.config.data_loader_workers,
        #                                     pin_memory=self.config.pin_memory,
        #                                     drop_last=True)
        # self.test_loader = data.DataLoader(test_set,
        #                                     batch_size=self.config.batch_size,
        #                                     shuffle=False,
        #                                     num_workers=self.config.data_loader_workers,
        #                                     pin_memory=self.config.pin_memory,
        #                                     drop_last=True)
        data_set = City_Dataset(data_path=self.config.cityscapes_path,
                                list_path='/city_list',
                                split=split,
                                base_size=self.config.base_size,
                                crop_size=self.config.crop_size)

        if split == "train" or split == "trainval":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.config.batch_size,
                                               shuffle=True,
                                               num_workers=self.config.data_loader_workers,
                                               pin_memory=self.config.pin_memory,
                                               drop_last=True)
        elif split =="val" or split == "test":
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.config.batch_size,
                                               shuffle=False,
                                               num_workers=self.config.data_loader_workers,
                                               pin_memory=self.config.pin_memory,
                                               drop_last=True)
        else:
            raise Warning("split must be train/val/trainavl/test")



        self.iterations_num = (len(data_set) + self.config.batch_size) // self.config.batch_size





if __name__ == '__main__':
    data=scipy.io.loadmat('/data/linhua/VOCdevkit/BSD/dataset/cls/2008_003846.mat')
    print(data['GTcls']["Segmentation"][0,0])
    print(np.array([[(1,2,3)]]).shape)
    print(np.array([[np.array(1), np.array(2), np.array(3)]]).shape)