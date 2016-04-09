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
# a.sgd2(gamma=0.001,decay_method=0) # 学习速率较小，收敛慢
# a.sgd2(gamma=0.01,decay_method=0) # 学习速率较小，收敛慢
# a.sgd2(gamma=0.04,decay_method=0) # 默认学习速率较小，train：1.063, test: 0.987
# a.sgd2(gamma=0.1,decay_method=0) # 学习速率适中，比默认值收敛到更好的值，train: 1.046, test: 1.001
# a.sgd2(gamma=0.1,decay_method=0) # 学习速率适中，比默认值收敛到更好的值
# a.sgd2(gamma=0.2,decay_method=0) # 学习速率过大，快速回弹
# a.sgd2(gamma=0.3,decay_method=0) # 学习速率过大，快速回弹
# a.sgd2(gamma=0.1,decay=1000) # 学习速率适中，比默认值收敛到更好的值
# a.sgd2(gamma=0.04,decay_method=1) # 自定义自适应学习速率，初始学习速率太小，train：1.258, test: 1.257
# a.sgd2(gamma=0.08,decay_method=1) # 自定义自适应学习速率，初始学习速率适中，train：1.046, test: 0.991
# a.sgd2(gamma=0.1,decay_method=1) # 自定义自适应学习速率，初始学习速率适中，train：1.046, test: 0.998
# a.sgd2(gamma=0.4,decay_method=1) # 自定义自适应学习速率，初始学习速率太大，train：2.339, test: 2.339
# a.sgd2(gamma=0.004,decay_method=2) # adagrad，初始学习速率太小，train：1.688, test: 1.611
# a.sgd2(gamma=0.008,decay_method=2) # adagrad，初始学习速率太小，train：1.623, test: 1.542
# a.sgd2(gamma=0.01,decay_method=2) # adagrad，初始学习速率适中，train：1.557, test: 1.508
# a.sgd2(gamma=0.012,decay_method=2) # adagrad，初始学习速率太小，train：1.620, test: 1.574
# a.sgd2(gamma=0.015,decay_method=2) # adagrad，初始学习速率太小，train：1.719, test: 1.745
# a.sgd2(gamma=0.02,decay_method=2) # adagrad，初始学习速率太大，train：1.856, test: 1.851
# a.sgd2(gamma=0.04,decay_method=2) # adagrad，初始学习速率太大，train：2.185, test: 2.185
# a.sgd2(gamma=0.4,decay_method=2) # adagrad，初始学习速率太大，train：2.329, test: 2.341
# a.sgd2(gamma=0.04,decay_method=3) # adadelta，初始学习速率适中，train：1.063, test: 0.991
 a.sgd2(gamma=0.08,decay_method=3) # adadelta，初始学习速率适中，train：1.040, test: 0.986
# a.sgd2(gamma=0.1,decay_method=3) # adadelta，初始学习速率适中，train：1.042, test: 0.999
# a.sgd2(gamma=0.15,decay_method=3) # adadelta，初始学习速率太大，train：2.163, test: 2.165
# a.mbgd(steps=1,gamma=0.04,Lambda=0.15,batch_size=1,decay_a=10) # 模拟online learning的方式，train: 1.064 test: 0.990
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
# a.mbgd_adadelta(steps=20,gamma=0.04,Lambda=0.15,batch_size=256,decay_a=1,decay_enable=True)
# a.mbgd2(steps=1,gamma=0.04,Lambda=0.15,batch_size=1,decay_a=10) # 模拟online learning的方式，train: 1.064 test: 0.990
# a.ftrl(alpha=0.5,beta=1.0,lambda1=1.0,lambda2=1.0) # train: 0.973, test: 0.968，z=zeros(k,1)，此时学习到的user和item隐式主题向量为0
# a.ftrl(alpha=1.0,beta=1.0,lambda1=0.0,lambda2=1.0) # train 1.202, test: 1.230，随机初始化z=random()/sqrt(k)，学习速率过大，反弹
# a.ftrl(alpha=0.5,beta=1.0,lambda1=0.0,lambda2=1.0) # train: 0.980, test: 0.979，随机初始化z=random()/sqrt(k)
# a.ftrl(alpha=0.1,beta=1.0,lambda1=0.0,lambda2=1.0) # train 0.991, test: 0.980，随机初始化z=random(k,1)/sqrt(k)
# a.ftrl(alpha=0.05,beta=1.0,lambda1=0.0,lambda2=1.0) # train 1.024, test: 1.018，随机初始化z=random(k,1)/sqrt(k)
# a.ftrl3(alpha=1.0,beta=1.0,lambda1=1.0,lambda2=1.0)
a.test(test_X)

# a.ftrl2(alpha=0.5,beta=1.0,lambda1=1.0,lambda2=1.0,gamma=1.0) # test: 0.968
# a.test2(test_X,gamma=1.0)
