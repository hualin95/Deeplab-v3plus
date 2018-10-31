# A *Higher Performance* Pytorch Implementation of DeepLab V3 Plus 

## Introduction
This repo is an (re-)implementation of [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) in PyTorch for semantic image segmentation on the PASCAL VOC dataset. And this repo has a higher mIoU of 79.19% than the result of paper which is 78.85%.

## Requirements
Python(3.6) and Pytorch(0.4.1) is necessary before running the scripts.
To install the required python packages(expect PyTorch), run
```python
pip install -r requirements.txt
```

## Datasets
To train and validate the network, this repo use the augmented PASCAL VOC 2012 dataset which contains 10582 images for training and 1449 images for validation. To use the dataset, you can download the  PASCAL VOC training/validation data (2GB tar file) [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and download the SegmentationClassAug from [dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Baidu Netdisk](https://pan.baidu.com/s/1x6HteW_Rs-XXa8Y_vOvwxg)


## Training
Before training, you should clone this repo:
```python
git clone git@github.com:hualin95/Deeplab-v3plus.git
```

You can begin training by running the train.py.
```python
#training
cd Deeplab-v3plus-master/tools/   
python train.py
```
You are expected to achieve PA:94.77%, MPA:88.48%, MIoU:79.19%, FWIoU:90.53% on the validation.
```python
#Monitoring
tensorboard --logdir=runs/ --port=80
```
![](https://github.com/hualin95/Deeplab-v3plus/blob/master/data/tensorboardX.png)

## Performance
VOC2012: after 30k iterations with a batch size of 16.

| Backbone | train OS|eval OS| MS | mIoU paper| mIoU repo|
| :--------| :------:|:-----:|:--:|:---------:|:--------:|
| Resnet101|16       |16     |No  |78.85%     |79.19%    |

## TODO
- [x] Resnet as Network Backbone
- [x] Implement depthwise separable convolutions
- [x] Multi-GPU support
- [ ] Model pretrained on MS-COCO
- [ ] Xception as Network Backbone