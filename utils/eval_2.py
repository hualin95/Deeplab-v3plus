# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 13:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : eval_2.py
# @Software: PyCharm

import os
import time
import numpy as np
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')



class Eval():
    def __init__(self, gt_image, pre_imgae, num_class):
        self.gt_image = gt_image
        self.pre_image = pre_imgae
        self.num_class = num_class

        # assert the size of two images are same
        assert self.gt_image.shape == self.pre_image.shape

        self.accuracy_matrix = self.__generate_matrix(self.gt_image, self.pre_image)



    def Pixel_Accuracy(self):
        if np.sum(self.accuracy_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.accuracy_matrix).sum() / self.accuracy_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self):
        MPA = np.diag(self.accuracy_matrix) / self.accuracy_matrix.sum(axis=1)
        MPA = np.nanmean(MPA)

        return MPA

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.accuracy_matrix) / (
                    np.sum(self.accuracy_matrix, axis=1) + np.sum(self.accuracy_matrix, axis=0) -
                    np.diag(self.accuracy_matrix))
        MIoU = np.nanmean(MIoU)

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        FWIoU = np.multiply(np.sum(self.accuracy_matrix, axis=1), np.diag(self.accuracy_matrix))
        FWIoU = FWIoU / (np.sum(self.accuracy_matrix, axis=1) + np.sum(self.accuracy_matrix, axis=0) -
                         np.diag(self.accuracy_matrix))
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.accuracy_matrix)

        return FWIoU


    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, ):



class Eval_from_filepath():
    def __init__(self, val_list_filepath, gt_filepath, pre_filepath, num_class):
        self.val_list_filepath = val_list_filepath
        self.gt_filepath = gt_filepath
        self.pre_filepath = pre_filepath
        self.items = [id.strip() for id in open(val_list_filepath)]
        self.num_class = num_class

    def __getitem__(self):
        evaluation = {}
        evaluation['PA'] = []
        evaluation['MPA'] = []
        evaluation['MIoU'] = []
        evaluation['FWIoU'] = []
        for id in self.items:
            gt_image_path = os.path.join(self.gt_filepath, "{}.jpg".format(id))
            pre_image_path = os.path.join(self.pre_filepath, "{}.png".format(id))
            gt_image = Image.open(gt_image_path).convert('P')
            pre_image = Image.open(pre_image_path).convert('P')
            eval_one_pair = Eval(gt_image, pre_image, self.num_class)
            evaluation['PA'].append(eval_one_pair.Pixel_Accuracy())
            evaluation['MPA'].append(eval_one_pair.Mean_Pixel_Accuracy())
            evaluation['MIoU'].append(eval_one_pair.Mean_Intersection_over_Union())
            evaluation['FWIoU'].append(eval_one_pair.Frequency_Weighted_Intersection_over_Union())

        return evaluation




if __name__ == "__main__":
    gt_image = Image.open('F:/projects/Deeplab v3plus/imgaes/gt/2007_000129.png')
    pre_image = Image.open('F:/projects/Deeplab v3plus/imgaes/pred/2007_000129.png')

    gt_image = np.array(gt_image)
    pre_image = np.array(pre_image)


    time_start = time.time()
    metric = Eval(gt_image, pre_image, 21)
    print(metric.Pixel_Accuracy(), metric.Mean_Pixel_Accuracy(), metric.Mean_Intersection_over_Union(),
          metric.Frequency_Weighted_Intersection_over_Union())
    time_end = time.time()
    total = time_end - time_start
    print(total)

