#!/user/bin/python
#-*- coding:utf-8 -*-

from __future__ import division
import time

import numpy as np
import scipy as sp
from numpy.random import random
from pylab import *

class SVD:
    # 初始化：模型参数、训练样本
    # X: np.array 二维数组，训练样本[uid,mid,rat]
    # k: int, 隐主题维度数
    def __init__(self,X,k=20):
        self.X=np.array(X)
        self.k=k
        self.ave=np.mean(self.X[:,2])
        print "the input data size is ",self.X.shape
        self.bi={}
        self.bu={}
        self.qi={}
        self.pu={}
        self.movie_user={}
        self.user_movie={}
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]
            mid=self.X[i][1]
            rat=self.X[i][2]
            self.movie_user.setdefault(mid,{})
            self.user_movie.setdefault(uid,{})
            self.movie_user[mid][uid]=rat
            self.user_movie[uid][mid]=rat
            self.bi.setdefault(mid,0)
            self.bu.setdefault(uid,0)
            self.qi.setdefault(mid,random((self.k,1))/10*(np.sqrt(self.k)))
            self.pu.setdefault(uid,random((self.k,1))/10*(np.sqrt(self.k)))

    # 预测
    def pred(self,uid,mid):
        self.bi.setdefault(mid,0)
        self.bu.setdefault(uid,0)
        self.qi.setdefault(mid,np.zeros((self.k,1)))
        self.pu.setdefault(uid,np.zeros((self.k,1)))
        if (self.qi[mid] is None):
            self.qi[mid]=np.zeros((self.k,1))
        if (self.pu[uid] is None):
            self.pu[uid]=np.zeros((self.k,1))
        ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum(self.qi[mid]*self.pu[uid])
        if ans>5:
            return 5
        elif ans<1:
            return 1
        return ans

    def pred2(self,uid,mid,gamma):
        self.bi.setdefault(mid,0)
        self.bu.setdefault(uid,0)
        self.qi.setdefault(mid,np.zeros((self.k,1)))
        self.pu.setdefault(uid,np.zeros((self.k,1)))
        if (self.qi[mid] is None):
            self.qi[mid]=np.zeros((self.k,1))
        if (self.pu[uid] is None):
            self.pu[uid]=np.zeros((self.k,1))
        ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum((self.qi[mid]+gamma)*(self.pu[uid]+gamma))
        if ans>5:
            return 5
        elif ans<1:
            return 1
        return ans

    # SGD
    # 特点：
    #	1. 一般能够收敛到全局最优（凸目标函数）或局部最优（非凸目标函数），但存在收敛到马鞍点的问题
    #	2. 收敛速度较快，但是不是很稳定
    # 注：
    #   1. 同BGD、mini-batch GD，需要在训练集上反复迭代，以收敛到最佳点附近
    #   2. 多次迭代需要配合衰减的学习速率，以收敛到更接近最优解的点
    #   3. 每次迭代需要shuffle训练样本，克服局部相关性
    def sgd(self,steps=20,gamma=0.04,Lambda=0.15):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        iter_rmse = np.zeros((steps, 2))
        for step in range(steps):
            print 'the ',step,'-th  step is running'
            rmse_sum=0.0
            kk=np.random.permutation(self.X.shape[0])
            batch = 1024
            for j in range(self.X.shape[0]):
                i=kk[j]
                uid=self.X[i][0]
                mid=self.X[i][1]
                rat=self.X[i][2]
                eui=rat-self.pred(uid,mid)
                rmse_sum+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])
                temp=self.qi[mid]
                self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid])
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])
            gamma=gamma*0.93
            print "the rmse of this step on train data is ",np.sqrt(rmse_sum/self.X.shape[0])
            iter_rmse[step] = [step, np.sqrt(rmse_sum/self.X.shape[0])]
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        plot(iter_rmse[:,0],iter_rmse[:,1])
        show()

    # SGD
    #   1. 一遍SGD的情况，
    #   2. 引入了自适应学习速率
    #   1) 自定义自适应学习速率：gamma = gamma*sqrt(a/(t+a))，a是衰减因子
    #   2) adagrad：优点是不存在原生SGD的马鞍点问题，收敛速度慢，常跟downpour sgd搭配使用。trick: gamma的值比原生SGD的最优gamma值大一个量级。实验结果收敛效果并不及固定学习速率的SGD，应该是收敛速度慢的原因，收敛速度慢主要原因应该是对每个w独立的学习速率
    #   2) adadelta：优点是不存在原生SGD的马鞍点问题，而且收敛速度快于adagrad，而且不需要指定初始学习速率，完全自适应。实验的收敛效果跟SGD相当
    #   注：adagrad和adadelta按wi独立学习速率的特点有助于并行化
    def sgd2(self,steps=20,gamma=0.04,Lambda=0.15,decay_method=0,decay=300):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        gamma_init = gamma
        rmse_sum=0.0
        rmse_lst=[]
        kk=np.random.permutation(self.X.shape[0]) # avoid locality
        bi_info = {} # (grad_bi,delta_bi,n)
        bu_info = {} # (grad_bu,delta_bu,n)
        qi_info = {} # (grad_qi,delta_qi,n)
        pu_info = {} # (grad_pu,delta_pu,n)
        for j in range(self.X.shape[0]):	# rmse 0.985
            # print "gamma: ",gamma
            i=kk[j]
            # i = j
            uid=self.X[i][0]
            mid=self.X[i][1]
            rat=self.X[i][2]
            eui=rat-self.pred(uid,mid)
            rmse_sum+=eui**2
            if decay_method == 0: # 固定学习速率
                self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                temp=self.qi[mid]
                self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid])
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])
            elif decay_method == 1: # 自定义自适应学习速率
                self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                temp=self.qi[mid]
                self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid])
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])
                gamma=gamma_init*np.sqrt(decay/(decay+j))
            elif decay_method == 2: # adagrad
                gamma_bi = gamma/np.sqrt(bi_info[mid][0] if mid in bi_info and bi_info[mid][0]>0 else 1.0)
                grad_bi = (eui-Lambda*self.bi[mid])
                delta_bi = gamma_bi*grad_bi
                self.bi[mid]+=delta_bi
                gamma_bu = gamma/np.sqrt(bu_info[uid][0] if uid in bu_info and bu_info[uid][0]>0 else 1.0)
                grad_bu = (eui-Lambda*self.bu[uid])
                delta_bu = gamma_bu*grad_bu
                self.bu[uid]+=delta_bu
                temp=self.qi[mid]
                gamma_qi = gamma/np.sqrt(qi_info[mid][0] if mid in qi_info else np.ones((self.k,1)))
                grad_qi = (eui*self.pu[uid]-Lambda*self.qi[mid])
                delta_qi = gamma_qi*grad_qi
                self.qi[mid]+=delta_qi
                gamma_pu = gamma/np.sqrt(pu_info[uid][0] if uid in pu_info else np.ones((self.k,1)))
                grad_pu = (eui*temp-Lambda*self.pu[uid])
                delta_pu = gamma_pu*grad_pu
                self.pu[uid]+=delta_pu
                bi_info[mid]=(bi_info[mid] if mid in bi_info else np.array([0,0,0]))+np.array([grad_bi**2,delta_bi**2,1])
                bu_info[uid]=(bu_info[uid] if uid in bu_info else np.array([0,0,0]))+np.array([grad_bu**2,delta_bu**2,1])
                qi_info[mid]=(qi_info[mid] if mid in qi_info else np.array([0,0,0]))+np.array([grad_qi**2,delta_qi**2,1])
                pu_info[uid]=(pu_info[uid] if uid in pu_info else np.array([0,0,0]))+np.array([grad_pu**2,delta_pu**2,1])
            elif decay_method == 3: # adadelta
                gamma_bi = np.sqrt(bi_info[mid][1])/np.sqrt(bi_info[mid][0]) if mid in bi_info and bi_info[mid][0]>0 else gamma
                grad_bi = (eui-Lambda*self.bi[mid])
                delta_bi = gamma_bi*grad_bi
                self.bi[mid]+=delta_bi
                gamma_bu = np.sqrt(bu_info[uid][1])/np.sqrt(bu_info[uid][0]) if uid in bu_info and bu_info[uid][0]>0 else gamma
                grad_bu = (eui-Lambda*self.bu[uid])
                delta_bu = gamma_bu*grad_bu
                self.bu[uid]+=delta_bu
                temp=self.qi[mid]
                gamma_qi = np.sqrt(qi_info[mid][1])/np.sqrt(qi_info[mid][0]) if mid in qi_info else gamma*np.ones((self.k,1))
                grad_qi = (eui*self.pu[uid]-Lambda*self.qi[mid])
                delta_qi = gamma_qi*grad_qi
                self.qi[mid]+=delta_qi
                gamma_pu = np.sqrt(pu_info[uid][1])/np.sqrt(pu_info[uid][0]) if uid in pu_info else gamma*np.ones((self.k,1))
                grad_pu = (eui*temp-Lambda*self.pu[uid])
                delta_pu = gamma_pu*grad_pu
                self.pu[uid]+=delta_pu
                bi_info[mid]=(bi_info[mid] if mid in bi_info else np.array([0,0,0]))+np.array([grad_bi**2,delta_bi**2,1])
                bu_info[uid]=(bu_info[uid] if uid in bu_info else np.array([0,0,0]))+np.array([grad_bu**2,delta_bu**2,1])
                qi_info[mid]=(qi_info[mid] if mid in qi_info else np.array([0,0,0]))+np.array([grad_qi**2,delta_qi**2,1])
                pu_info[uid]=(pu_info[uid] if uid in pu_info else np.array([0,0,0]))+np.array([grad_pu**2,delta_pu**2,1])
            if j%1000 == 0:
                rmse = np.sqrt(rmse_sum/(j+1))
                rmse_lst.append(rmse)
                print "the rmse of this step %d-th on train data is %f" % (j, rmse)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        print "the rmse of this step on train data is ",np.sqrt(rmse_sum/self.X.shape[0])
        # print "bi: ",self.bi
        # print "bu: ",self.bu
        # print "qi: ",self.qi
        # print "pu: ",self.pu
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    # BGD
    def bgd(self,steps=20,gamma=0.04,Lambda=0.15):
        start_time = time.time()
        batch_gd_update_once(self.X, gamma, Lambda)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)

    # Mini-batch训练
    # 优点：
    #	1. 减少参数更新的方差，比SGD更稳定收敛
    #	2. 可以充分利用高度优化的矩阵计算库
    # 	3. 并行求解可以减少更新参数的网络开销（SGD每次都更新参数，网络开销太大）
    # 经验：
    #	1. mini-batch的大小一般取值50~256
    def mbgd(self,steps=20,gamma=0.04,Lambda=0.15,batch_size=64,decay_a=10,decay_enable=False):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        rmse_lst = []
        gamma_init = gamma
        count = 0
        for step in range(steps):
            np.random.shuffle(self.X)
            rmse = 0.0
            for batch in self.get_batches(self.X, batch_size=batch_size):
                rmse += self.batch_gd_update_once(batch, gamma, Lambda)
                if count%10000 == 0:
                    print "rmse: ",rmse
                count += 1
            rmse = np.sqrt(rmse/self.X.shape[0])
            rmse_lst.append(rmse)
            print "the rmse of this step %d-th on train data is %f" % (step, rmse)
            if decay_enable:
                # gamma *= 0.93
                gamma = gamma_init*np.sqrt(decay_a/(decay_a+step))
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    # 每次Mini-batch后使用累计平均权重作为模型权重
    def mbgd2(self,steps=20,gamma=0.04,Lambda=0.15,batch_size=64,decay_a=10,decay_enable=False):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        rmse_lst = []
        gamma_init = gamma
        count = 0
        bi_sum = {}
        bu_sum = {}
        qi_sum = {}
        pu_sum = {}
        for step in range(steps):
            np.random.shuffle(self.X)
            rmse = 0.0
            for batch in self.get_batches(self.X, batch_size=batch_size):
                rmse += self.batch_gd_update_once(batch, gamma, Lambda)
                if count%10000 == 0:
                    print "rmse: ",rmse
                count += 1
                bi_sum = self.add_mul_dict_1d(bi_sum, self.bi)
                bu_sum = self.add_mul_dict_1d(bu_sum, self.bu)
                qi_sum = self.add_mul_dict_1d(qi_sum, self.qi)
                pu_sum = self.add_mul_dict_1d(pu_sum, self.pu)
            rmse = np.sqrt(rmse/self.X.shape[0])
            rmse_lst.append(rmse)
            print "the rmse of this step %d-th on train data is %f" % (step, rmse)
            if decay_enable:
                # gamma *= 0.93
                gamma = gamma_init*np.sqrt(decay_a/(decay_a+step))
        self.bi = self.add_mul_dict_1d({}, bi_sum, 1.0/count)
        self.bu = self.add_mul_dict_1d({}, bu_sum, 1.0/count)
        self.qi = self.add_mul_dict_1d({}, qi_sum, 1.0/count)
        self.pu = self.add_mul_dict_1d({}, pu_sum, 1.0/count)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    # 一维dict相加
    def add_mul_dict_1d(self, a1, a2, mul=1.0):
        res = {}
        for k in a2:
            res[k] = mul*(a2[k]+(0 if k not in a1 else a1[k]))
        return res

    # 二维dict相加
    def add_mul_dict_2d(self, a1, a2, mul=1.0):
        res = {}
        for k1 in a2:
            if k1 not in res:
                res[k1] = 0
            for k2 in a2[k1]:
                res[k1][k2] = mul*(a2[k1][k2]+(0 if (k1 not in a1 or k2 not in a1[k1]) else a1[k1][k2]))
        return res

    # 批量样本生成器
    def get_batches(self, X, batch_size=100):
        n = X.shape[0]
        for s in range(0, n, batch_size):
            e = s+batch_size if s+batch_size<n else n
            yield X[s:e]

    # 批量梯度下降
    def batch_gd_update_once(self, batch, gamma, Lambda):
        eui = {}
        uids = set()
        mids = set()
        rmse_sum = 0.0
        for i in range(batch.shape[0]):
            uid = batch[i][0]
            mid = batch[i][1]
            rat = batch[i][2]
            err = rat-self.pred(uid,mid)
            rmse_sum += err**2
            if uid not in eui:
                eui[uid] = {}
            eui[uid][mid] = err
            uids.add(uid)
            mids.add(mid)
        qi_tmp = self.qi
        # 更新bi、qi
        for mid in mids:
            # 更新正则项梯度
            self.bi[mid] -= gamma*Lambda*self.bi[mid]
            self.qi[mid] -= gamma*Lambda*self.qi[mid]
            # 更新损失项梯度
            for uid in uids:
                if uid in eui and mid in eui[uid]:
                    self.bi[mid] += gamma*eui[uid][mid]
                    self.qi[mid] += gamma*eui[uid][mid]*self.pu[uid]
        # 更新bu、pu
        for uid in uids:
            # 更新正则项梯度
            self.bu[uid] -= gamma*Lambda*self.bu[uid]
            self.pu[uid] -= gamma*Lambda*self.pu[uid]
            # 更新损失项梯度
            for mid in mids:
                if uid in eui and mid in eui[uid]:
                    self.bu[uid] += gamma*eui[uid][mid]
                    self.pu[uid] += gamma*eui[uid][mid]*qi_tmp[mid]
        # return np.sqrt(rmse_sum/batch.shape[0])
        return rmse_sum

    # Mini-batch + Adadelta自适应学习速率
    # Adadelta自适应学习速率优点：
    #   1. 收敛速度快
    #	2. 能够收敛到全局最优（凸目标函数）或局部最优（非凸目标函数），不存在收敛到马鞍点的问题
    # 注：实验上看并没有上述优点，实现有bug？
    def mbgd_adadelta(self,steps=20,gamma=0.04,Lambda=0.15,batch_size=64,decay_a=10,decay_enable=False):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        rmse_lst = []
        count = 0
        grad_delta_bi = {}
        grad_delta_bu = {}
        grad_delta_qi = {}
        grad_delta_pu = {}
        for step in range(steps):
            np.random.shuffle(self.X)
            rmse = 0.0
            for batch in self.get_batches(self.X, batch_size=batch_size):
                rmse += self.batch_gd_update_once_adadelta(batch, gamma, Lambda, grad_delta_bi, grad_delta_bu, grad_delta_qi, grad_delta_pu)
                if count%10000 == 0:
                    print "rmse: ",rmse
                count += 1
            rmse = np.sqrt(rmse/self.X.shape[0])
            rmse_lst.append(rmse)
            print "the rmse of this step %d-th on train data is %f" % (step, rmse)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    # 批量梯度下降+adadelta自适应梯度下降
    def batch_gd_update_once_adadelta(self, batch, gamma, Lambda, grad_delta_bi, grad_delta_bu, grad_delta_qi, grad_delta_pu):
        eui = {}
        uids = set()
        mids = set()
        rmse_sum = 0.0
        eps = 1e-8
        eta = 30
        for i in range(batch.shape[0]):
            uid = batch[i][0]
            mid = batch[i][1]
            rat = batch[i][2]
            err = rat-self.pred(uid,mid)
            rmse_sum += err**2
            if uid not in eui:
                eui[uid] = {}
            eui[uid][mid] = err
            uids.add(uid)
            mids.add(mid)
        qi_tmp = self.qi
        # 更新bi、qi
        for mid in mids:
            # 计算学习速率
            if mid not in grad_delta_bi:
                gamma_bi = abs(np.random.normal(0,1)/eta)
                grad_delta_bi[mid] = np.zeros(3, dtype='float')
            else:
                # print "grad_delta_bi[%s] shape: %s" % (mid, grad_delta_bi[mid].shape)
                # print mid
                # print grad_delta_bi[mid]
                gamma_bi = np.sqrt(grad_delta_bi[mid][1]+eps)/np.sqrt(grad_delta_bi[mid][0]+eps)
            if mid not in grad_delta_qi:
                gamma_qi = abs(np.random.normal(0,1)/eta)
                grad_delta_qi[mid] = np.array([np.zeros((self.k,1),dtype='float'),np.zeros((self.k,1),dtype='float'),1])
            else:
                gamma_qi = np.sqrt(grad_delta_qi[mid][1]+eps)/np.sqrt(grad_delta_qi[mid][0]+eps)
            # 更新正则项梯度
            grad_bi = Lambda*self.bi[mid]
            grad_qi = Lambda*self.qi[mid]
            # 更新损失项梯度
            for uid in uids:
                if uid in eui and mid in eui[uid]:
                    grad_bi -= eui[uid][mid]
                    grad_qi -= eui[uid][mid]*self.pu[uid]
            # 更新参数
            delta_bi = -gamma_bi*grad_bi
            delta_qi = -gamma_qi*grad_qi
            self.bi[mid] += delta_bi
            self.qi[mid] += delta_qi
            grad_delta_bi[mid] += (grad_bi**2, delta_bi**2, 1)
            grad_delta_qi[mid] += (grad_qi**2, delta_qi**2, 1)
        # 更新bu、pu
        for uid in uids:
            # 计算学习速率
            if uid not in grad_delta_bu:
                gamma_bu = abs(np.random.normal(0,1)/eta)
                grad_delta_bu[uid] = np.zeros(3, dtype='float')
            else:
                gamma_bu = np.sqrt(grad_delta_bu[uid][1]+eps)/np.sqrt(grad_delta_bu[uid][0]+eps)
            if uid not in grad_delta_pu:
                gamma_pu = abs(np.random.normal(0,1)/eta)
                grad_delta_pu[uid] = np.array([np.zeros((self.k,1),dtype='float'),np.zeros((self.k,1),dtype='float'),1])
            else:
                gamma_pu = np.sqrt(grad_delta_pu[uid][1]+eps)/np.sqrt(grad_delta_pu[uid][0]+eps)
            # 更新正则项梯度
            grad_bu = Lambda*self.bu[uid]
            grad_pu = Lambda*self.pu[uid]
            # 更新损失项梯度
            for mid in mids:
                if uid in eui and mid in eui[uid]:
                    grad_bu -= eui[uid][mid]
                    grad_pu -= eui[uid][mid]*qi_tmp[mid]
            # 更新参数
            delta_bu = -gamma_bu*grad_bu
            delta_pu = -gamma_pu*grad_pu
            self.bu[uid] += delta_bu
            self.pu[uid] += delta_pu
            grad_delta_bu[uid] += (grad_bu**2, delta_bu**2, 1)
            grad_delta_pu[uid] += (grad_pu**2, delta_pu**2, 1)
        # return np.sqrt(rmse_sum/batch.shape[0])
        return rmse_sum

    # FTRL
    # 优点：
    #   1. 只需要扫描一遍数据集就可以收敛到最优点
    #   2. 更容易求得稀疏解
    def ftrl(self,alpha=1.0,beta=1.0,lambda1=1.0,lambda2=1.0):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        np.random.shuffle(self.X)
        z_bi = {}
        n_bi = {}
        z_bu = {}
        n_bu = {}
        z_qi = {}
        n_qi = {}
        z_pu = {}
        n_pu = {}
        rmse_sum = 0
        rmse_lst = []
        for t in xrange(self.X.shape[0]):
            uid = self.X[t][0]
            mid = self.X[t][1]
            rat = self.X[t][2]
            # 更新bi
            z_bi.setdefault(mid, 0)
            n_bi.setdefault(mid, 0)
            self.bi[mid] = self.update_param(z_bi[mid],n_bi[mid],alpha,beta,lambda1,lambda2)
            # 更新bu
            z_bu.setdefault(uid, 0)
            n_bu.setdefault(uid, 0)
            self.bu[uid] = self.update_param(z_bu[uid],n_bu[uid],alpha,beta,lambda1,lambda2)
            # 更新qi
            z_qi.setdefault(mid, np.zeros((self.k,1)))
            # z_qi.setdefault(mid, np.random.random((self.k,1))/np.sqrt(self.k))
            n_qi.setdefault(mid, np.zeros((self.k,1)))
            self.qi[mid] = self.update_param(z_qi[mid],n_qi[mid],alpha,beta,lambda1,lambda2)
            # 更新pu
            z_pu.setdefault(uid, np.zeros((self.k,1)))
            # z_pu.setdefault(uid, np.random.random((self.k,1))/np.sqrt(self.k))
            n_pu.setdefault(uid, np.zeros((self.k,1)))
            self.pu[uid] = self.update_param(z_pu[uid],n_pu[uid],alpha,beta,lambda1,lambda2)
            # 预测
            eui = self.pred(uid,mid)-rat
            rmse_sum += eui**2
            if t%1000 == 0:
                rmse = np.sqrt(rmse_sum/(t+1))
                rmse_lst.append(rmse)
                print "the rmse of this step %d-th on train data is %f" % (t, rmse)
            # 更新z_bi,n_bi
            g_bi = eui
            z_bi[mid],n_bi[mid] = self.update_z_n(g_bi,z_bi[mid],n_bi[mid],self.bi[mid],alpha)
            # 更新z_bu,n_bu
            g_bu = eui
            z_bu[uid],n_bu[uid] = self.update_z_n(g_bu,z_bu[uid],n_bu[uid],self.bu[uid],alpha)
            # 更新z_qi,n_qi
            qi_tmp = self.qi[mid]
            g_qi = eui*(self.pu[uid])
            z_qi[mid],n_qi[mid] = self.update_z_n(g_qi,z_qi[mid],n_qi[mid],self.qi[mid],alpha)
            # 更新z_pu,n_pu
            g_pu = eui*(qi_tmp)
            z_pu[uid],n_pu[uid] = self.update_z_n(g_pu,z_pu[uid],n_pu[uid],self.pu[uid],alpha)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        # print "bi: ",self.bi
        # print "bu: ",self.bu
        # print "qi: ",self.qi
        # print "pu: ",self.pu
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    def ftrl2(self,alpha=1.0,beta=1.0,lambda1=1.0,lambda2=1.0,gamma=0.0):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        np.random.shuffle(self.X)
        z_bi = {}
        n_bi = {}
        z_bu = {}
        n_bu = {}
        z_qi = {}
        n_qi = {}
        z_pu = {}
        n_pu = {}
        rmse_sum = 0
        rmse_lst = []
        for t in xrange(self.X.shape[0]):
            uid = self.X[t][0]
            mid = self.X[t][1]
            rat = self.X[t][2]
            # 更新bi
            z_bi.setdefault(mid, 0)
            n_bi.setdefault(mid, 0)
            self.bi[mid] = self.update_param(z_bi[mid],n_bi[mid],alpha,beta,lambda1,lambda2)
            # 更新bu
            z_bu.setdefault(uid, 0)
            n_bu.setdefault(uid, 0)
            self.bu[uid] = self.update_param(z_bu[uid],n_bu[uid],alpha,beta,lambda1,lambda2)
            # 更新qi
            z_qi.setdefault(mid, np.zeros((self.k,1)))
            n_qi.setdefault(mid, np.zeros((self.k,1)))
            self.qi[mid] = self.update_param(z_qi[mid],n_qi[mid],alpha,beta,lambda1,lambda2)
            # 更新pu
            z_pu.setdefault(uid, np.zeros((self.k,1)))
            n_pu.setdefault(uid, np.zeros((self.k,1)))
            self.pu[uid] = self.update_param(z_pu[uid],n_pu[uid],alpha,beta,lambda1,lambda2)
            # 预测
            # eui = self.pred(uid,mid)-rat
            eui = self.pred2(uid,mid,gamma)-rat
            rmse_sum += eui**2
            if t%1000 == 0:
                rmse = np.sqrt(rmse_sum/(t+1))
                rmse_lst.append(rmse)
                print "the rmse of this step %d-th on train data is %f" % (t, rmse)
            # 更新z_bi,n_bi
            g_bi = eui
            z_bi[mid],n_bi[mid] = self.update_z_n(g_bi,z_bi[mid],n_bi[mid],self.bi[mid],alpha)
            # 更新z_bu,n_bu
            g_bu = eui
            z_bu[uid],n_bu[uid] = self.update_z_n(g_bu,z_bu[uid],n_bu[uid],self.bu[uid],alpha)
            # 更新z_qi,n_qi
            qi_tmp = self.qi[mid]
            g_qi = eui*(self.pu[uid]+gamma)
            z_qi[mid],n_qi[mid] = self.update_z_n(g_qi,z_qi[mid],n_qi[mid],self.qi[mid],alpha)
            # 更新z_pu,n_pu
            g_pu = eui*(qi_tmp+gamma)
            z_pu[uid],n_pu[uid] = self.update_z_n(g_pu,z_pu[uid],n_pu[uid],self.pu[uid],alpha)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        print "bi: ",self.bi
        print "bu: ",self.bu
        print "qi: ",self.qi
        print "pu: ",self.pu
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    def ftrl3(self,alpha=1.0,beta=1.0,lambda1=1.0,lambda2=1.0):
        print "the train data size is: %d" % self.X.shape[0]
        start_time = time.time()
        np.random.shuffle(self.X)
        z_bi = {}
        n_bi = {}
        z_bu = {}
        n_bu = {}
        z_qi = {}
        n_qi = {}
        z_pu = {}
        n_pu = {}
        rmse_sum = 0
        rmse_lst = []
        for t in xrange(self.X.shape[0]):
            uid = self.X[t][0]
            mid = self.X[t][1]
            rat = self.X[t][2]
            # 初始化
            z_bi.setdefault(mid, 0)
            n_bi.setdefault(mid, 0)
            z_bu.setdefault(uid, 0)
            n_bu.setdefault(uid, 0)
            z_qi.setdefault(mid, np.zeros((self.k,1)))
            n_qi.setdefault(mid, np.zeros((self.k,1)))
            z_pu.setdefault(uid, np.zeros((self.k,1)))
            n_pu.setdefault(uid, np.zeros((self.k,1)))
            # 预测
            eui = self.pred(uid,mid)-rat
            rmse_sum += eui**2
            if t%1000 == 0:
                rmse = np.sqrt(rmse_sum/(t+1))
                rmse_lst.append(rmse)
                print "the rmse of this step %d-th on train data is %f" % (t, rmse)
            # 更新z_bi,n_bi
            g_bi = eui
            z_bi[mid],n_bi[mid] = self.update_z_n(g_bi,z_bi[mid],n_bi[mid],self.bi[mid],alpha)
            # 更新z_bu,n_bu
            g_bu = eui
            z_bu[uid],n_bu[uid] = self.update_z_n(g_bu,z_bu[uid],n_bu[uid],self.bu[uid],alpha)
            # 更新z_qi,n_qi
            qi_tmp = self.qi[mid]
            g_qi = eui*self.pu[uid]
            z_qi[mid],n_qi[mid] = self.update_z_n(g_qi,z_qi[mid],n_qi[mid],self.qi[mid],alpha)
            # 更新z_pu,n_pu
            g_pu = eui*qi_tmp
            z_pu[uid],n_pu[uid] = self.update_z_n(g_pu,z_pu[uid],n_pu[uid],self.pu[uid],alpha)
            # 更新bi
            self.bi[mid] = self.update_param(z_bi[mid],n_bi[mid],alpha,beta,lambda1,lambda2)
            # 更新bu
            self.bu[uid] = self.update_param(z_bu[uid],n_bu[uid],alpha,beta,lambda1,lambda2)
            # 更新qi
            self.qi[mid] = self.update_param(z_qi[mid],n_qi[mid],alpha,beta,lambda1,lambda2)
            # 更新pu
            self.pu[uid] = self.update_param(z_pu[uid],n_pu[uid],alpha,beta,lambda1,lambda2)
        end_time = time.time()
        print "time cost: %f" % (end_time-start_time)
        print "bi: ",self.bi
        print "bu: ",self.bu
        print "qi: ",self.qi
        print "pu: ",self.pu
        plot(range(len(rmse_lst)), rmse_lst)
        show()

    def update_param(self,z,n,alpha,beta,lambda1,lambda2):
        w = -1.0/((beta+np.sqrt(n))/alpha+lambda2)*(z-np.sign(z)*lambda1)
        w = w*(abs(z)>lambda1) # w=0 if |z|<=lambda1, for sparsity
        return w

    def update_z_n(self,grad,z,n,w,alpha):
        sigma = 1.0/alpha*(np.sqrt(n+grad**2)-np.sqrt(n))
        z += grad-sigma*w
        n += grad**2
        return z,n

    def ftrl_core(self,w,X,alpha,beta,lambda1,lambda2):
        # self.ftrl_core(self.bi,self.X,alpha,beta,lambda1,lambda2)
        # self.ftrl_core(self.bu,self.X,alpha,beta,lambda1,lambda2)
        # self.ftrl_core(self.qi,self.X,alpha,beta,lambda1,lambda2)
        # self.ftrl_core(self.pu,self.X,alpha,beta,lambda1,lambda2)
        # z = {}
        # n = {}
        # for i in xrange(X.shape[0]):
        #     uid = X[i][0]
        #     mid = X[i][1]
        #     rat = X[i][2]
        # for k in w:
        pass


    # 测试
    def test(self,test_X):
        output=[]
        sums=0
        test_X=np.array(test_X)
        print "the test data size is ",test_X.shape
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            output.append(pre)
            #print pre,test_X[i][2]
            sums+=(pre-test_X[i][2])**2
        rmse=np.sqrt(sums/test_X.shape[0])
        print "the rmse on test data is ",rmse
        return output

    def test2(self,test_X,gamma=1.0):
        output=[]
        sums=0
        test_X=np.array(test_X)
        print "the test data size is ",test_X.shape
        for i in range(test_X.shape[0]):
            pre=self.pred2(test_X[i][0],test_X[i][1],gamma)
            output.append(pre)
            #print pre,test_X[i][2]
            sums+=(pre-test_X[i][2])**2
        rmse=np.sqrt(sums/test_X.shape[0])
        print "the rmse on test data is ",rmse
        return output
