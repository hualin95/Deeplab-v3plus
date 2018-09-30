# -*- coding: utf-8 -*-
# @Time    : 2018/9/29 21:43
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : eval.py
# @Software: PyCharm

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


        self.gt_mask = self.__generate_mask(self.gt_image, self.num_class)
        self.pre_mask = self.__generate_mask(self.pre_image, self.num_class)
        self.accuracy_matrix= self.__generate_matrix(self.gt_mask, self.pre_mask)

    def Pixel_Accuracy(self):
        if np.sum(self.accuracy_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.accuracy_matrix).sum()/self.accuracy_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self):
        MPA = np.diag(self.accuracy_matrix)/self.accuracy_matrix.sum(axis=1)
        MPA = np.nanmean(MPA)

        return MPA

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.accuracy_matrix)/(np.sum(self.accuracy_matrix, axis=1)+np.sum(self.accuracy_matrix,axis=0)-
                                              np.diag(self.accuracy_matrix))
        MIoU = np.nanmean(MIoU)

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        FWIoU = np.multiply(np.sum(self.accuracy_matrix,axis=1),np.diag(self.accuracy_matrix))
        FWIoU = FWIoU/(np.sum(self.accuracy_matrix, axis=1)+np.sum(self.accuracy_matrix,axis=0)-
                       np.diag(self.accuracy_matrix))
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i))/np.sum(self.accuracy_matrix)

        return FWIoU

    def __generate_mask(self, image, class_num):
        mask = np.zeros((class_num, image.shape[0], image.shape[1]))

        for i in range(class_num):
            mask[i, :, :] = (image==i)

        return mask

    def __generate_matrix(self, gt_mask, pre_mask):
        assert gt_mask.shape == pre_mask.shape

        accuracy_matrix = np.zeros((self.num_class,)*2)

        for i in range(self.num_class):
            for j in range(self.num_class):
                accuracy_matrix[i, j] = np.sum(np.logical_and(gt_mask[i,:,:], pre_mask[j,:,:]))

        return accuracy_matrix



if __name__ =="__main__":
    gt_image = Image.open('F:/projects/Deeplab v3plus/imgaes/gt/2007_000129.png')
    pre_image = Image.open('F:/projects/Deeplab v3plus/imgaes/pred/2007_000129.png')

    gt_image = np.array(gt_image)
    pre_image = np.array(pre_image)

    metric = Eval(gt_image, pre_image, 21)
    print(metric.Pixel_Accuracy(), metric.Mean_Pixel_Accuracy(), metric.Mean_Intersection_over_Union(),
          metric.Frequency_Weighted_Intersection_over_Union())

