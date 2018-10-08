# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 15:48
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train.py
# @Software: PyCharm

import os
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from tqdm import tqdm
from configparser import ConfigParser

import sys
sys.path.append(os.path.abspath('..'))

from utils.data_utils import calculate_weigths_labels
from utils.eval_2 import Eval
from graphs.models.decoder import Decoder
from datasets.Voc_Dataset import VOCDataLoader
from configs.global_config import cfg


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--loss_weight', default=False)
arg_parser.add_argument('--num_classes', default=21)
arg_parser.add_argument('--lr', default=0.05)
arg_parser.add_argument('--imagenet_pretrained', default=True)
arg_parser.add_argument('--data_root_path', default="/data/linhua/VOCdevkit/")
arg_parser.add_argument('--result_filepath', default="/data/linhua/VOCdevkit/VOC2012/Results/")
arg_parser.add_argument('--store_result', default=False)
arg_parser.add_argument('--checkpoint_dir', default=os.path.abspath('..')+"/checkpoints/")


config = ConfigParser()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logger.txt')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

class Trainer():
    def __init__(self, args, config, cuda=None):
        self.args = args
        self.config = config
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.best_MIou = 0
        self.current_epoch = 0
        self.epoch_num = self.config.epoch_num
        self.current_iter = 0
        self.val_list_filepath = os.path.join(args.data_root_path, 'VOC2012/ImageSets/Segmentation/val.txt')
        self.gt_filepath = os.path.join(args.data_root_path, 'VOC2012/SegmentationClass/')
        self.pre_filepath = os.path.join(args.data_root_path, 'VOC2012/JPEGImages/')
        self.Eval = Eval(self.config.num_classes)
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
        self.model = nn.DataParallel(self.model)
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
        # set TensorboardX
        writer = SummaryWriter()

        # display config details
        logger.info("Global configuration as follows:")
        logger.info(self.config)

        # choose cuda
        if self.cuda:
            # torch.cuda.set_device(4)
            current_device = torch.cuda.current_device()
            logger.info("This model will run on", torch.cuda.get_device_name(current_device))
        else:
            logger.info("This model will run on CPU")

        # train
        self.train()

        writer.close()

    def train(self):
        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.config.epoch_num)):
            self.current_epoch = epoch
            self.scheduler.step(epoch)
            self.train_one_epoch()

            # validate
            PA, MPA, MIoU, FWIoU = self.validate()
            logger.info("PA:{}, MPA:{}, MIou:{}, FWIoU:{}".format(PA, MPA, MIoU, FWIoU))

            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
            self.save_checkpoint(is_best, str(self.current_epoch)+'.pth')
            # writer.add_scalar('PA', PA)
            # print(PA)



    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.dataloader.train_loader, total=self.dataloader.train_iterations,
                          desc="Train Epoch-{}-".format(self.current_epoch+1))
        # Set the model to be in training mode (for batchnorm)
        train_loss = []
        self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, _ in tqdm_epoch:
            # y.to(torch.long)
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            # model
            pred = self.model(x)
            y = torch.squeeze(y, 1)
            # loss
            cur_loss = self.loss(pred, y)

            # print(cur_loss)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer

            cur_loss.backward()
            self.optimizer.step()
            train_loss.append(cur_loss.item())
            self.current_iter += 1
            if batch_idx % self.config.batch_save ==0:
                tqdm.write("The loss of epoch{}-batch-{}:{}".format(self.current_epoch, batch_idx, cur_loss.item()))
            batch_idx += 1

        loss = sum(train_loss)/len(train_loss)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, loss))
        tqdm_epoch.close()

    def validate(self):
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.valid_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            val_loss = []
            self.model.eval()

            for x, y, id in tqdm_batch:
                # y.to(torch.long)
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)
                y = torch.squeeze(y, 1)
                cur_loss = self.loss(pred, y)
                # print(cur_loss)
                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during training...')

                val_loss.append(cur_loss.item())

                argpred = torch.argmax(pred,dim=1)
                self.Eval.add_batch(y.cpu().numpy(), argpred.cpu().numpy())

                if self.args.store_result == 'True':
                    for i in range(len(id)):
                        result = Image.fromarray(y[i])
                        result.save(self.args.result_filepath + id[i] + '.png', mdoe='P')

            loss = sum(val_loss) / len(val_loss)
            logger.info("The average loss of val loss:{}".format(loss))


            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, is_best, filename=None):
        """
        Save checkpoint if a new best is achieved
        :param state:
        :param is_best:
        :param filepath:
        :return:
        """
        filename = os.path.join(self.args.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if is_best:
            logger.info("=>saving a new best checkpoint.ing...")
            torch.save(state, filename)
        else:
            logger.info("=> The MIoU of val does't improve.")

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename








if __name__ == '__main__':
    args = arg_parser.parse_args()
    config.read("../configs/deeplab.cfg", encoding="UTF-8")
    agent = Trainer(args=args, config=cfg, cuda=True)
    agent.main()
