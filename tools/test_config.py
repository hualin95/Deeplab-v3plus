# -*- coding: utf-8 -*-
# @Time    : 2018/9/28 14:15
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : test_config.py
# @Software: PyCharm

from configparser import ConfigParser

config = ConfigParser()
config.read("../configs/deeplab.cfg")

print(config.sections())
print(config.options('data'))
print(config['DEFAULT']['name'])
print(config['data']['num_classes'])
print(config.items('model'))
print(config.name)
