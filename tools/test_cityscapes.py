# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 10:10
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : test_cityscapes.py
# @Software: PyCharm


import os
import pprint
import logging
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import sys
sys.path.append(os.path.abspath('..'))
from graphs.models.sync_batchnorm.replicate import patch_replication_callback
from utils.data_utils import calculate_weigths_labels
from utils.eval import Eval
from graphs.models.decoder import DeepLab
from datasets.Voc_Dataset import VOCDataLoader
from configs.global_config import cfg

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--loss_weight', type=str2bool, default=False)
arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True)
arg_parser.add_argument('--data_root_path', default="/data/linhua/VOCdevkit/")
arg_parser.add_argument('--result_filepath', default="/data/linhua/VOCdevkit/VOC2012/Results/")
arg_parser.add_argument('--store_result', type=str2bool, default=True)
arg_parser.add_argument('--checkpoint_dir', default=os.path.abspath('..')+"/checkpoints/")
arg_parser.add_argument('--saved_checkpoint_file')
arg_parser.add_argument('--pretrained', type=str2bool, default=False)
arg_parser.add_argument('--store_checkpoint_name', default="voc2012_no_class_weight_sync_bn_26")
arg_parser.add_argument('--freeze_bn', type=str2bool, default=False)
arg_parser.add_argument('--bn_momentum', type=float, default=0.1)
arg_parser.add_argument('--lr', type=float, default=0.007)
arg_parser.add_argument('--iter_max', type=int, default=30000)
arg_parser.add_argument('--gpu', type=str, default="4,5,6,7")
arg_parser.add_argument('--output_stride', type=int, default=16)
arg_parser.add_argument('--dataset', type=str, default='voc2012_aug')



logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logger_with_weight_aug_sync.txt')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

class Trainer():
    def __init__(self, args, config, cuda=None):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
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

        # loss definition
        if args.loss_weight:
            classes_weights_path = os.path.join(self.config.classes_weight, self.args.dataset + 'classes_weights_log.npy')
            print(classes_weights_path)
            if not os.path.isfile(classes_weights_path):
                logger.info('calculating class weights...')
                calculate_weigths_labels(self.config)
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
                             class_num=self.config.num_classes,
                             pretrained=self.args.imagenet_pretrained,
                             bn_momentum=self.args.bn_momentum,
                             freeze_bn=self.args.freeze_bn)
        self.model = nn.DataParallel(self.model, device_ids=range(4))
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
            momentum=self.config.momentum,
            # dampening=self.config.dampening,
            weight_decay=self.config.weight_decay,
            # nesterov=self.config.nesterov
        )
        # dataloader
        self.dataloader = VOCDataLoader(self.args, self.config)

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
            self.load_checkpoint(self.args.saved_checkpoint_file)

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

            PA, MPA, MIoU, FWIoU = self.validate()
            self.writer.add_scalar('PA', PA, self.current_epoch)
            self.writer.add_scalar('MPA', MPA, self.current_epoch)
            self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)



            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
            self.save_checkpoint(is_best, self.args.store_checkpoint_name)

            # writer.add_scalar('PA', PA)
            # print(PA)



    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.dataloader.train_loader, total=self.dataloader.train_iterations,
                          desc="Train Epoch-{}-".format(self.current_epoch+1))
        logger.info("Training one epoch...")
        self.Eval.reset()
        # Set the model to be in training mode (for batchnorm and dropout)

        train_loss = []
        preds = []
        lab = []
        self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.config.poly_power,
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

            if batch_idx % self.config.batch_save == 0:
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
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
        #
        if key == "10x":
            for m in model.named_modules():
                if "encoder" in m[0] or "decoder" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p


    def poly_lr_scheduler(self, optimizer, init_lr, iter, max_iter, power):
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        optimizer.param_groups[1]["lr"] = 10 * new_lr








if __name__ == '__main__':
    args = arg_parser.parse_args()
    agent = Trainer(args=args, config=cfg, cuda=True)
    agent.main()
