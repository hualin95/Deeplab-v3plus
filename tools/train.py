# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 15:48
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train.py
# @Software: PyCharm

import os
import glog as log
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from tqdm import tqdm, trange
from configparser import ConfigParser

import sys
sys.path.append(os.path.abspath('..'))

from utils.data_utils import calculate_weigths_labels
from graphs.models.decoder import Decoder
from datasets.Voc_Dataset import VOCDataLoader
from configs.global_config import cfg


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--loss_weight', default=False)
arg_parser.add_argument('--num_classes', default=21)
arg_parser.add_argument('--lr', default=0.05)
arg_parser.add_argument('--imagenet_pretrained', default=True)

config = ConfigParser()

class Trainer():
    def __init__(self, args, config, cuda=None):
        self.args = args
        self.config = config
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_epoch = 0
        self.epoch_num = self.config.epoch_num
        self.current_iter = 0
        # loss definition
        if args.loss_weight:
            if not os.path.isfile(self.config.classes_weight):
                calculate_weigths_labels(self.config)
            class_weights = np.load(self.config.class_weights)
            weight = torch.from_numpy(class_weights.astype(np.float32))
        else:
            weight = None

        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        self.loss.to(self.device)

        # model
        self.model = Decoder(self.config.num_classes, pretrained=args.imagenet_pretrained).to(self.device)
        # self.model = nn.DataParallel(self.model)
        # self.model.cuda()

        # optimizer
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                          #lr=self.config['model']['lr'],
                                          lr=args.lr,
                                          momentum=self.config.momentum,
                                          dampening=self.config.dampening,
                                          weight_decay=self.config.weight_decay,
                                          nesterov=self.config.nesterov)

        # dataloader
        self.dataloader = VOCDataLoader(self.config)

        # lr_scheduler
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=self.config.step_size,
                                             gamma=self.config.gamma)





    def main(self):
        writer = SummaryWriter()
        if self.cuda:
            # torch.cuda.set_device(4)
            current_device = torch.cuda.current_device()
            print("This model will run on", torch.cuda.get_device_name(current_device))
        else:
            print("This model will run on CPU")

        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.config.epoch_num)):
            self.current_epoch = epoch
            self.scheduler.step(epoch)
            train_loss = self.train_one_epoch()

            writer.add_scalar('train', train_loss, epoch)
        writer.close()
            # train_epoch.close()


    def train_one_epoch(self):
        tqdm_batch = tqdm(self.dataloader.train_loader, total=self.dataloader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch+1))
        # Set the model to be in training mode (for batchnorm)
        train_loss = []
        self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y in tqdm_batch:
            # y.to(torch.long)
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            # model
            pred = self.model(x)
            # print(pred.size())
            # loss
            y = torch.squeeze(y, dim=1)
            # print(y.size())
            cur_loss = self.loss(pred, y)
            # print(cur_loss)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer

            cur_loss.backward()
            self.optimizer.step()
            train_loss.append(cur_loss.data.cpu().numpy())
            self.current_iter += 1
            if batch_idx % self.config.save ==0:
                tqdm.write("The loss of epoch:{}-batch-{}".format(self.current_epoch, batch_idx))
            batch_idx += 1

        loss = sum(train_loss)/len(train_loss)
        tqdm.write("The average loss of epoch-{}-:{}".format(self.current_epoch, loss))
        tqdm_batch.close()
        return loss


if __name__ == '__main__':
    args = arg_parser.parse_args()
    config.read("../configs/deeplab.cfg", encoding="UTF-8")
    agent = Trainer(args=args, config=cfg, cuda=True)
    agent.main()
