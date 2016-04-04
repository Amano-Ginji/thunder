#!/user/bin/python
#-*- coding:utf-8 -*-

from __future__ import division

import numpy as np
import scipy as sp

from src.env import *
from src.mf import *

def read_data():
    train = open(data_dir+'/ml-100k/u1.base').read().splitlines()
    test = open(data_dir+'/ml-100k/u1.test').read().splitlines()
    train_X = []
    test_X = []
    for line in train:
        p = line.split('	');
        train_X.append([int(p[0]),int(p[1]),int(p[2])])
    for line in test:
        p = line.split('	');
        test_X.append([int(p[0]),int(p[1]),int(p[2])])
    return train_X, test_X

train_X, test_X = read_data()
print np.array(train_X).shape, np.array(test_X).shape

a = SVD(train_X, 30)
# a.sgd()
# a.sgd(200)
# a.sgd2(200,gamma=0.1)
# a.sgd2(gamma=0.001,decay_enable=False) # 学习速率较少，收敛慢
# a.sgd2(gamma=0.01,decay_enable=False) # 学习速率较少，收敛慢
# a.sgd2(gamma=0.04,decay_enable=False) # 默认学习速率较少
# a.sgd2(gamma=0.1,decay_enable=False) # 学习速率适中，比默认值收敛到更好的值
# a.sgd2(gamma=0.1,decay_enable=False) # 学习速率适中，比默认值收敛到更好的值
# a.sgd2(gamma=0.2,decay_enable=False) # 学习速率过大，快速回弹
# a.sgd2(gamma=0.3,decay_enable=False) # 学习速率过大，快速回弹
# a.sgd2(gamma=0.1,decay=1000) # 学习速率适中，比默认值收敛到更好的值
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=1,decay_a=10) # train: 0.820 test: 0.934
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=64,decay_a=10) # train: 0.803 test: 0.931
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=10) # train: 0.747 test: 0.926
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=10,decay_enable=True) # train: 0.753 test: 0.922, 有一些过拟，可以尝试调正则项优化
# a.mbgd(steps=20,gamma=0.04,Lambda=0.3,batch_size=256,decay_a=10,decay_enable=True) # train: 0.919 test: 0.955
# a.mbgd(steps=20,gamma=0.04,Lambda=0.2,batch_size=256,decay_a=10,decay_enable=True) # train: 0.846 test: 0.930
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=1,decay_enable=True) # train: 0.794 test: 0.926
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=50,decay_enable=True) # train: 0.748 test: 0.924
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=100,decay_enable=True) # train: 0.747 test: 0.925
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=1,decay_enable=True) # train: 0.794 test: 0.923
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=1024,decay_a=10) # train: 0.622 test: 0.949
# a.mbgd(steps=20,gamma=0.04,Lambda=0.15,batch_size=80000,decay_a=10) # 等价于全量梯度下降
a.mbgd_adadelta(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=1,decay_enable=True)
a.test(test_X)

