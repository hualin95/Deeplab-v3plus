# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 19:52
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : generate_list.py
# @Software: PyCharm


import os

train_path = r"F:\Datasets\Cityscapes\gtFine_trainvaltest\gtFine\train"
val_path = r"F:\Datasets\Cityscapes\gtFine_trainvaltest\gtFine\val"
test_path = r"F:\Datasets\Cityscapes\gtFine_trainvaltest\gtFine\test"

train_list = r"F:\Datasets\Cityscapes\train.txt"
val_list = r"F:\Datasets\Cityscapes\val.txt"
test_list = r"F:\Datasets\Cityscapes\test.txt"

f = open(train_list, "w")
train_l = []
val_l = []
test_l = []
for root, dirs, _ in os.walk(train_path):
    for dir in dirs:
        dir = os.path.join(root, dir)
        files = os.listdir(dir)
        for file in files:
            if file[-12:-4] == 'labelIds':
                file_name = "train_"+file[:-20]+"\n"
                train_l.append(file_name)
f.writelines(train_l)

f = open(val_list, "w")
for root, dirs, _ in os.walk(val_path):
    for dir in dirs:
        dir = os.path.join(root, dir)
        files = os.listdir(dir)
        for file in files:
            if file[-12:-4] == 'labelIds':
                file_name = "val_"+file[:-20]+"\n"
                val_l.append(file_name)
f.writelines(val_l)

f = open(test_list, "w")
for root, dirs, _ in os.walk(test_path):
    for dir in dirs:
        dir = os.path.join(root, dir)
        files = os.listdir(dir)
        for file in files:
            if file[-12:-4] == 'labelIds':
                file_name = "test_"+file[:-20]+"\n"
                test_l.append(file_name)
f.writelines(test_l)






