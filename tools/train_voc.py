# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 15:48
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train_voc.py
# @Software: PyCharm

import os
import pprint
import logging
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from math import ceil
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter

import sys
sys.path.append(os.path.abspath('..'))
from libs.modules.sync_batchnorm.replicate import patch_replication_callback
from libs.utils.data_utils import calculate_weigths_labels
from libs.utils import Eval
from libs.models.decoder import DeepLab
from libs.datasets.Voc_Dataset import VOCDataLoader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class Trainer():
    def __init__(self, args, cuda=None):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_MIoU = 0
        self.best_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0

        # set TensorboardX
        self.writer = SummaryWriter()

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        if self.args.loss_weight_file is not None:
            classes_weights_path = os.path.join(self.args.loss_weights_dir, self.args.loss_weight_file)
            print(classes_weights_path)
            if not os.path.isfile(classes_weights_path):
                logger.info('calculating class weights...')
                calculate_weigths_labels(self.args)
            class_weights = np.load(classes_weights_path)
            pprint.pprint(class_weights)
            weight = torch.from_numpy(class_weights.astype(np.float32))
            logger.info('loading class weights successfully!')
        else:
            weight = None

        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        self.loss.to(self.device)

        # model
        self.model = DeepLab(output_stride=self.args.output_stride,
                             class_num=self.args.num_classes,
                             pretrained=self.args.imagenet_pretrained and self.args.pretrained_ckpt_file==None,
                             bn_momentum=self.args.bn_momentum,
                             freeze_bn=self.args.freeze_bn)
        self.model = nn.DataParallel(self.model, device_ids=range(ceil(len(self.args.gpu)/2)))
        patch_replication_callback(self.model)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            params=[
                {
                    "params": self.get_params(self.model.module, key="1x"),
                    "lr": self.args.lr,
                },
                {
                    "params": self.get_params(self.model.module, key="10x"),
                    "lr": 10 * self.args.lr,
                },
            ],
            momentum=self.args.momentum,
            # dampening=self.args.dampening,
            weight_decay=self.args.weight_decay,
            # nesterov=self.args.nesterov
        )
        # dataloader
        self.dataloader = VOCDataLoader(self.args)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.train_iterations)

    def main(self):
        # display args details
        logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            logger.info("{:16} {}".format(key, val))

        # choose cuda
        if self.cuda:
            # torch.cuda.set_device(4)
            current_device = torch.cuda.current_device()
            logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            logger.info("This model will run on CPU")

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        # train
        self.train()

        self.writer.close()

    def train(self):
        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.epoch_num)):
            self.current_epoch = epoch
            # self.scheduler.step(epoch)
            self.train_one_epoch()

            # validate
            if self.args.validation == True:
                PA, MPA, MIoU, FWIoU = self.validate()
                self.writer.add_scalar('PA', PA, self.current_epoch)
                self.writer.add_scalar('MPA', MPA, self.current_epoch)
                self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
                self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)

                self.current_MIoU = MIoU
                is_best = MIoU > self.best_MIou
                if is_best:
                    self.best_MIou = MIoU
                self.save_checkpoint(is_best, train_id+'best.pth')

        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.current_MIoU
        }
        logger.info("=>saving the final checkpoint...")
        torch.save(state, train_id + 'final.pth')

    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.dataloader.train_loader, total=self.dataloader.train_iterations,
                          desc="Train Epoch-{}-".format(self.current_epoch+1))
        logger.info("Training one epoch...")
        self.Eval.reset()
        # Set the model to be in training mode (for batchnorm and dropout)

        train_loss = []
        self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.current_iter >= self.args.iter_max:
                logger.info("iteration arrive {}!".format(self.args.iter_max))
                break
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            self.writer.add_scalar('learning_rate_10x', self.optimizer.param_groups[1]["lr"], self.current_iter)

            # y.to(torch.long)
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            self.optimizer.zero_grad()

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

            if batch_idx % 50 == 0:
                logger.info("The train loss of epoch{}-batch-{}:{}".format(self.current_epoch,
                                                                           batch_idx, cur_loss.item()))
            batch_idx += 1

            self.current_iter += 1

            # print(cur_loss)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            self.Eval.add_batch(label, argpred)

        PA = self.Eval.Pixel_Accuracy()
        MPA = self.Eval.Mean_Pixel_Accuracy()
        MIoU = self.Eval.Mean_Intersection_over_Union()
        FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        logger.info('Epoch:{}, train PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                       MIoU, FWIoU))


        tr_loss = sum(train_loss)/len(train_loss)
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))
        tqdm_epoch.close()

    def validate(self):
        logger.info('validating one epoch...')
        self.Eval.reset()
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
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

            logger.info('Epoch:{}, validation PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                          MIoU, FWIoU))
            v_loss = sum(val_loss) / len(val_loss)
            logger.info("The average loss of val loss:{}".format(v_loss))
            self.writer.add_scalar('val_loss', v_loss, self.current_epoch)

            # logger.info(score)
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

            # self.current_epoch = checkpoint['epoch']
            # self.current_iter = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_MIou = checkpoint['best_MIou']

            logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {},MIoU:{})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration'],
                          checkpoint['best_MIou']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            logger.info("**First time to train**")

    def get_params(self, model, key):
        # For Dilated CNN
        if key == "1x":
            for m in model.named_modules():
                if "Resnet101" in m[0]:
                    for p in m[1].parameters():
                        yield p
        #
        if key == "10x":
            for m in model.named_modules():
                if "encoder" in m[0] or "decoder" in m[0]:
                    for p in m[1].parameters():
                        yield p


    def poly_lr_scheduler(self, optimizer, init_lr, iter, max_iter, power):
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        optimizer.param_groups[1]["lr"] = 10 * new_lr





if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    arg_parser = argparse.ArgumentParser()

    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default="/data/linhua/VOCdevkit/",
                            help="the root path of dataset")
    arg_parser.add_argument('--checkpoint_dir', default=os.path.abspath('..') + "/checkpoints/",
                            help="the path of ckpt file")
    arg_parser.add_argument('--result_filepath', default="/data/linhua/VOCdevkit/VOC2012/Results/",
                            help="the filepath where mask store")
    arg_parser.add_argument('--loss_weights_dir', default="/data/linhua/VOCdevkit/pretrained_weights/")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='resnet101',
                            help="backbone of encoder")
    arg_parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16],
                            help="choose from 8 or 16")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply iamgenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--save_ckpt_file', type=str2bool, default=True,
                            help="whether to save trained checkpoint file ")
    arg_parser.add_argument('--store_result_mask', type=str2bool, default=True,
                            help="whether store mask after val or test")
    arg_parser.add_argument('--loss_weight_file', type=str, default=None,
                            help="the filename of weights for loss function")
    arg_parser.add_argument('--validation', type=str2bool, default=True,
                            help="whether to val after each train epoch")

    # train related arguments
    arg_parser.add_argument('--gpu', type=str, default="1,3",
                            help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                            help='input batch size')

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='voc2012', type=str,
                            choices=['voc2012', 'voc2012_aug', 'cityscapes'],
                            help='dataset choice')
    arg_parser.add_argument('--base_size', default=513, type=int,
                            help='crop size of image')
    arg_parser.add_argument('--crop_size', default=513, type=int,
                            help='base size of image')
    arg_parser.add_argument('--num_classes', default=21, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=16, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train',
                            help="choose from train/val/test/trainval")

    # optimization related arguments

    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")

    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--dampening', type=float, default=0)
    arg_parser.add_argument('--nesterov', type=str2bool, default=True)
    arg_parser.add_argument('--weight_decay', type=float, default=4e-5)

    arg_parser.add_argument('--lr', type=float, default=0.007,
                            help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=30000,
                            help="the maxinum of iteration")
    arg_parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")
    args = arg_parser.parse_args()

    args.batch_size = args.batch_size_per_gpu * ceil(len(args.gpu) / 2)

    train_id = str(args.backbone) + '_' + str(args.output_stride)
    train_id += '_iamgenet_pre-' + str(args.imagenet_pretrained)
    train_id += '_ckpt_file-' + str(args.pretrained_ckpt_file)
    train_id += '_loss_weight_file-' + str(args.loss_weight_file)
    train_id += '_batch_size-' + str(args.batch_size)
    train_id += '_base_size-' + str(args.base_size)
    train_id += '_crop_size-' + str(args.crop_size)
    train_id += '_split-' + str(args.split)
    train_id += '_lr-' + str(args.lr)
    train_id += '_iter_max-' + str(args.iter_max)

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(train_id+'.txt')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    agent = Trainer(args=args, cuda=True)
    agent.main()
