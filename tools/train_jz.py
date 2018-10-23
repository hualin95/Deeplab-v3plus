# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 9:29
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train_jz.py
# @Software: PyCharm

import os
import pprint
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import gluoncv
import mxnet as mx

from tqdm import tqdm
from configparser import ConfigParser

import sys
sys.path.append(os.path.abspath('..'))

from utils.data_utils import calculate_weigths_labels
from utils.eval_2 import Eval
from utils.eval_3 import scores
from graphs.models.decoder import DeepLab, DeepLab_2
from graphs.models.resnet101 import DeepLabv3_plus
from graphs.models.deeplabv3plus import DeepLabV3Plus
from graphs.models.deeplabv2 import Res_Deeplab
from graphs.models.model_jz import DeepLabv3_plus
from datasets.Voc_Dataset import VOCDataLoader
from configs.global_config import cfg


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--loss_weight', default=False)
arg_parser.add_argument('--num_classes', default=21)
# arg_parser.add_argument('--lr', default=0.05)
arg_parser.add_argument('--imagenet_pretrained', default=True)
arg_parser.add_argument('--data_root_path', default="/data/linhua/VOCdevkit/")
arg_parser.add_argument('--result_filepath', default="/data/linhua/VOCdevkit/VOC2012/Results/")
arg_parser.add_argument('--store_result', default=True)
arg_parser.add_argument('--checkpoint_dir', default=os.path.abspath('..')+"/checkpoints/")
arg_parser.add_argument('--pretrained', default=False)


config = ConfigParser()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logger1.txt')
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

        self.writer = SummaryWriter()

        # path definition
        self.val_list_filepath = os.path.join(args.data_root_path, 'VOC2012/ImageSets/Segmentation/val.txt')
        self.gt_filepath = os.path.join(args.data_root_path, 'VOC2012/SegmentationClass/')
        self.pre_filepath = os.path.join(args.data_root_path, 'VOC2012/JPEGImages/')

        # Metric definition
        self.Eval = Eval(self.config.num_classes)

        self.metric = gluoncv.utils.metrics.SegmentationMetric(self.config.num_classes)

        self.lr = self.config.lr

        # loss definition
        if args.loss_weight:
            classes_weights_path = os.path.join(self.config.classes_weight, self.config.dataset + 'classes_weights.npy')
            print(classes_weights_path)
            if not os.path.isfile(classes_weights_path):
                logger.info('calculating class weights...')
                calculate_weigths_labels(self.config)
            class_weights = np.load(classes_weights_path)
            pprint.pprint(class_weights)
            weight = torch.from_numpy(class_weights.astype(np.float32))
        else:
            weight = None

        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        self.loss.to(self.device)

        # model
        self.model = DeepLabv3_plus(3, 21, 16, True)
        # self.model = DeepLab(16, class_num=21, pretrained=True)
        # self.model = DeepLabV3Plus(n_classes=21,
        #                            n_blocks=[3, 4, 23, 3],
        #                            pyramids=[6, 12, 18],
        #                            grids=[1, 2, 4],
        #                            output_stride=16,)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)



        # optimizer
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                          lr=self.config.lr,
                                          momentum=self.config.momentum,
                                          # dampening=self.config.dampening,
                                          weight_decay=self.config.weight_decay)
                                          # nesterov=self.config.nesterov)
        # self.optimizer = torch.optim.SGD(
        #     # cf lr_mult and decay_mult in train.prototxt
        #     params=[
        #         {
        #             "params": self.get_params(self.model.module, key="1x"),
        #             "lr": self.config.lr,
        #             "weight_decay": self.config.weight_decay,
        #         },
        #         {
        #             "params": self.get_params(self.model.module, key="10x"),
        #             "lr": 10 * self.config.lr,
        #             "weight_decay": self.config.weight_decay,
        #         },
        #         {
        #             "params": self.get_params(self.model.module, key="20x"),
        #             "lr": 20 * self.config.lr,
        #             "weight_decay": 0.0,
        #         },
        #     ],
        #     momentum=self.config.momentum,
        #     # dampening=self.config.dampening,
        #     weight_decay=self.config.weight_decay,
        #     nesterov=self.config.nesterov
        # )
        # dataloader
        self.dataloader = VOCDataLoader(self.config)


        # lr_scheduler
        # lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.epoch_num)), 0.9)
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        # self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
        #                                      step_size=self.config.step_size,
        #                                      gamma=self.config.gamma)

        #





    def main(self):
        # set TensorboardX


        # display config details
        logger.info("Global configuration as follows:")
        pprint.pprint(self.config)
        pprint.pprint(self.args)

        # choose cuda
        if self.cuda:
            # torch.cuda.set_device(4)
            current_device = torch.cuda.current_device()
            logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            logger.info("This model will run on CPU")

        # load pretrained checkpoint
        if self.args.pretrained:
            self.load_checkpoint(self.config.checkpoint_file)

        # train
        self.train()

        self.writer.close()

    def train(self):
        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.config.epoch_num)):
            self.current_epoch = epoch
            # self.scheduler.step(epoch)
            self.train_one_epoch()

            # validate
            # score = self.validate_3()
            # PA, MPA, MIoU, FWIoU = score.values()
            # PA, MPA, MIoU, FWIoU = self.validate()
            PA1, MPA1, MIoU1, FWIoU1, score, PA3, MIoU3 = self.validate_2()
            PA2, MPA2, MIoU2, FWIoU2 = score.values()
            # logger.info("PA:{}, MPA:{}, MIou:{}, FWIoU:{}".format(PA, MPA, MIoU, FWIoU))
            #
            self.writer.add_scalar('PA1', PA1, self.current_epoch)
            self.writer.add_scalar('MPA1', MPA1, self.current_epoch)
            self.writer.add_scalar('MIoU1', MIoU1, self.current_epoch)
            self.writer.add_scalar('FWIoU1', FWIoU1, self.current_epoch)

            self.writer.add_scalar('PA3', PA2, self.current_epoch)
            self.writer.add_scalar('MPA3', MPA2, self.current_epoch)
            self.writer.add_scalar('MIoU3', MIoU2, self.current_epoch)
            self.writer.add_scalar('FWIoU3', FWIoU2, self.current_epoch)

            self.writer.add_scalar('PA3', PA3, self.current_epoch)
            # self.writer.add_scalar('MPA', MPA1, self.current_epoch)
            self.writer.add_scalar('MIoU3', MIoU3, self.current_epoch)
            # self.writer.add_scalar('FWIoU', FWIoU1, self.current_epoch)


            is_best = MIoU3 > self.best_MIou
            if is_best:
                self.best_MIou = MIoU3
            self.save_checkpoint(is_best, 'bestper.pth')

            if self.current_iter >= self.config.iter_max:
                logger.info("iteration arrive 30000!")
                break
            # writer.add_scalar('PA', PA)
            # print(PA)



    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.dataloader.train_loader, total=self.dataloader.train_iterations,
                          desc="Train Epoch-{}-".format(self.current_epoch+1))
        logger.info("Training one epoch...")
        self.Eval.reset()
        self.metric.reset()
        # Set the model to be in training mode (for batchnorm)

        train_loss = []
        preds = []
        lab = []
        self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, _ in tqdm_epoch:
            # self.poly_lr_scheduler(
            #     optimizer=self.optimizer,
            #     init_lr=self.config.lr,
            #     iter=self.current_iter,
            #     lr_decay_iter=self.config.lr_decay,
            #     max_iter=self.config.iter_max,
            #     power=self.config.poly_power,
            # )
            # self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            # self.writer.add_scalar('learning_rate_10x', self.optimizer.param_groups[1]["lr"], self.current_iter)
            # self.writer.add_scalar('learning_rate_20x', self.optimizer.param_groups[2]["lr"], self.current_iter)

            # y.to(torch.long)
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            self.lr = self.adjust_learning_rate(self.optimizer, self.current_iter)
            # logger.info("current learning rate is:{}".format(self.lr))
            # model
            pred = self.model(x)
            # logger.info("pre:{}".format(pred.data.cpu().numpy()))
            y = torch.squeeze(y, 1)
            # logger.info("y:{}".format(y.cpu().numpy()))
            # pred_s = F.softmax(pred, dim=1)
            # loss
            cur_loss = self.loss(pred, y)

            # optimizer

            cur_loss.backward()
            self.optimizer.step()

            train_loss.append(cur_loss.item())

            if batch_idx % self.config.batch_save == 0:
                logger.info("The loss of epoch{}-batch-{}:{}".format(self.current_epoch, batch_idx, cur_loss.item()))
            batch_idx += 1

            self.current_iter += 1

            # print(cur_loss)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            tt = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            mask = (argpred > 0) & (argpred < 21)
            # logger.info(mask.sum())

            outputs = mx.nd.array(pred)
            targets = mx.nd.array(tt)

            self.Eval.add_batch(tt, argpred)
            self.metric.update(targets, outputs)
            lab += list(tt)
            preds += list(argpred)

        PA = self.Eval.Pixel_Accuracy()
        MPA = self.Eval.Mean_Pixel_Accuracy()
        MIoU = self.Eval.Mean_Intersection_over_Union()
        FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        score = scores(lab, preds, n_class=21)
        PA2, MPA2, MIoU2, FWIoU2 = score.values()
        pixAcc, mIoU = self.metric.get()

        logger.info('Epoch:{}, validation PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                       MIoU, FWIoU))
        logger.info('Epoch:{}, validation PA2:{}, MPA2:{}, MIoU2:{}, FWIoU2:{}'.format(self.current_epoch, PA2,
                                                                                       MPA2, MIoU2, FWIoU2))
        logger.info('Epoch:{}, validation PA3:{}, MIoU3:{}'.format(self.current_epoch, pixAcc, mIoU))


        logger.info("current learning rate is:{}".format(self.lr))
        tr_loss = sum(train_loss)/len(train_loss)
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))
        tqdm_epoch.close()

    def validate(self):
        self.Eval.reset()
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

                if self.args.store_result == True and self.current_epoch == 20:
                    for i in range(len(id)):
                        result = Image.fromarray(np.asarray(argpred, dtype=np.uint8)[i], mode='P')
                        # logger.info("before:{}".format(result.mode))
                        result = result.convert("RGB")
                        # logger.info("after:{}".format(result.mode))
                        # logger.info("shape:{}".format(result.getpixel((1,1))))
                        result.save(self.args.result_filepath + id[i] + '.png')

            v_loss = sum(val_loss) / len(val_loss)
            logger.info("The average loss of val loss:{}".format(v_loss))
            self.writer.add_scalar('val_loss', v_loss, self.current_epoch)

            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def validate_2(self):
        logger.info('validating one epoch...')
        self.Eval.reset()
        self.metric.reset()

        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.valid_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            val_loss = []
            preds = []
            lab = []
            self.model.eval()

            for x, y, id in tqdm_batch:
                # y.to(torch.long)
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)
                y = torch.squeeze(y, 1)

                cur_loss = self.loss(pred, y)
                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during validating...')
                val_loss.append(cur_loss.item())

                # if self.args.store_result == True and self.current_epoch == 20:
                #     for i in range(len(id)):
                #         result = Image.fromarray(np.asarray(argpred, dtype=np.uint8)[i], mode='P')
                #         # logger.info("before:{}".format(result.mode))
                #         result = result.convert("RGB")
                #         # logger.info("after:{}".format(result.mode))
                #         # logger.info("shape:{}".format(result.getpixel((1,1))))
                #         result.save(self.args.result_filepath + id[i] + '.png')

                pred = pred.data.cpu().numpy()
                tt = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                outputs = mx.nd.array(pred)
                targets = mx.nd.array(tt)

                self.Eval.add_batch(tt, argpred)
                self.metric.update(targets, outputs)
                lab += list(tt)
                preds += list(argpred)





            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

            score = scores(lab, preds, n_class=21)
            PA2, MPA2, MIoU2, FWIoU2 = score.values()
            pixAcc, mIoU = self.metric.get()

            logger.info('Epoch:{}, validation PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                          MIoU, FWIoU))
            logger.info('Epoch:{}, validation PA2:{}, MPA2:{}, MIoU2:{}, FWIoU2:{}'.format(self.current_epoch, PA2,
                                                                                           MPA2, MIoU2, FWIoU2))
            logger.info('Epoch:{}, validation PA3:{}, MIoU3:{}'.format(self.current_epoch, pixAcc, mIoU))

            v_loss = sum(val_loss) / len(val_loss)
            logger.info("The average loss of val loss:{}".format(v_loss))
            self.writer.add_scalar('val_loss', v_loss, self.current_epoch)

            # logger.info(score)
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU, score, pixAcc, mIoU

    def validate_3(self):
        logger.info('validating one epoch...')
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.valid_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            val_loss = []
            preds = []
            lab = []
            self.model.eval()

            for x, y, id in tqdm_batch:
                # y.to(torch.long)
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)
                y = torch.squeeze(y, 1)
                # pred_s = F.softmax(pred, dim = 1)
                cur_loss = self.loss(pred, y)
                # print(cur_loss)
                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during validating...')

                val_loss.append(cur_loss.item())

                pred = pred.data.cpu().numpy()
                argpred = np.argmax(pred, axis=1)
                lab += list (y.cpu().numpy())
                preds += list (argpred)

                if self.args.store_result == True and self.current_epoch == 20:
                    for i in range(len(id)):
                        result = Image.fromarray(np.asarray(argpred, dtype=np.uint8)[i], mode='P')
                        # logger.info("before:{}".format(result.mode))
                        result = result.convert("RGB")
                        # logger.info("after:{}".format(result.mode))
                        # logger.info("shape:{}".format(result.getpixel((1,1))))
                        result.save(self.args.result_filepath + id[i] + '.png')

            v_loss = sum(val_loss) / len(val_loss)
            logger.info("The average loss of val loss:{}".format(v_loss))
            self.writer.add_scalar('val_loss', v_loss, self.current_epoch)

            score = scores(lab, preds, n_class=21)
            # logger.info(score)
            tqdm_batch.close()

        return score

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
            'best_MIou':self.best_MIou
        }
        if is_best:
            logger.info("=>saving a new best checkpoint...")
            torch.save(state, filename)
        else:
            logger.info("=> The MIoU of val does't improve.")

    def load_checkpoint(self, filename):
        filename = os.path.join(self.args.checkpoint_dir, filename)
        try:
            logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_MIou = checkpoint['best_MIou']

            logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {},MIoU:{})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration'],
                          checkpoint['best_MIou']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            logger.info("**First time to train**")

    # def get_params(self, model, key):
    #     # For Dilated CNN
    #     if key == "1x":
    #         for m in model.named_modules():
    #             if "layer" in m[0]:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         yield p
    #     # For conv weight in the ASPP module
    #     if key == "10x":
    #         for m in model.named_modules():
    #             if "aspp" in m[0] :
    #                 if isinstance(m[1], nn.Conv2d):
    #                     yield m[1].weight
    #     # For conv bias in the ASPP module
    #     if key == "20x":
    #         for m in model.named_modules():
    #             if "aspp" in m[0] :
    #                 if isinstance(m[1], nn.Conv2d):
    #                     yield m[1].bias
    #
    # def poly_lr_scheduler(self, optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    #     # if iter % lr_decay_iter or iter > max_iter:
    #     #     return None
    #     new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    #     optimizer.param_groups[0]["lr"] = new_lr
    #     optimizer.param_groups[1]["lr"] = 10 * new_lr
    #     optimizer.param_groups[2]["lr"] = 20 * new_lr

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate(self, optimizer, i_iter):
        lr = self.lr_poly(self.config.lr, i_iter, self.config.iter_max, self.config.poly_power)
        optimizer.param_groups[0]['lr'] = lr
        return lr







if __name__ == '__main__':
    args = arg_parser.parse_args()
    agent = Trainer(args=args, config=cfg, cuda=True)
    agent.main()
