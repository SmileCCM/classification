#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import division
from sklearn.svm import SVC
from sklearn import metrics
from Tools import readbunchobj
import time

import numpy as np

stat = time.time()

# 导入训练集
trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
'''
# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
clf = SVC(kernel='rbf', probability=True).fit(train_set.tdm, train_set.label)
# 预测分类结果
predicted = clf.predict(test_set.tdm)
'''
'''
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
'''
'''
print("预测完毕!!!")
end = time.time()
print("预测时间：", end-stat)
# 计算分类精度：

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))


metrics_result(test_set.label, predicted)
'''
"""
Created on Sun Jul 29 17:15:25 2018
@author: rd
"""

"""
This dataset is part of MNIST dataset,but there is only 3 classes,
classes = {0:'0',1:'1',2:'2'},and images are compressed to 14*14 
pixels and stored in a matrix with the corresponding label, at the 
end the shape of the data matrix is 
num_of_images x 14*14(pixels)+1(lable)
"""
def load_data(split_ratio):
    tmp = np.load("data216x197.npy")
    data = tmp[:,:-1]
    label = tmp[:,-1]
    mean_data = np.mean(data, axis=0)
    train_data = data[int(split_ratio*data.shape[0]):]-mean_data
    train_label = label[int(split_ratio*data.shape[0]):]
    test_data = data[:int(split_ratio*data.shape[0])]-mean_data
    test_label = label[:int(split_ratio*data.shape[0])]
    return train_data, train_label, test_data, test_label

"""compute the hingle loss without using vector operation,
While dealing with a huge dataset,this will have low efficiency
X's shape [n,14*14+1],Y's shape [n,],W's shape [num_class,14*14+1]"""
def lossAndGradNaive(X,Y,W,reg):
    dW = np.zeros(W.shape)
    loss = 0.0
    num_class = W.shape[0]
    num_X = X.shape[0]
    for i in range(num_X):
        scores = np.dot(W, X[i])
        cur_scores = scores[int(Y[i])]
        for j in range(num_class):
            if j == Y[i]:
                continue
            margin = scores[j]-cur_scores+1
            if margin > 0:
                loss += margin
                dW[j,:] += X[i]
                dW[int(Y[i]),:]-=X[i]
    loss/=num_X
    dW/=num_X
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    return loss,dW

def lossAndGradVector(X,Y,W,reg):
    dW = np.zeros(W.shape)
    N = X.shape[0]
    Y_= X.dot(W.T)
    margin = Y_-Y_[range(N), Y.astype(int)].reshape([-1, 1])+1.0
    margin[range(N), Y.astype(int)] = 0.0
    margin = (margin > 0)*margin
    loss = 0.0
    loss += np.sum(margin)/N
    loss += reg*np.sum(W*W)
    """For one data,the X[Y[i]] has to be substracted several times"""
    countsX = (margin > 0).astype(int)
    countsX[range(N), Y.astype(int)] = -np.sum(countsX, axis=1)
    dW += np.dot(countsX.T, X)/N+2*reg*W
    return loss, dW

def predict(X,W):
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    Y_ = np.dot(X, W.T)
    Y_pre = np.argmax(Y_, axis=1)
    return Y_pre

def accuracy(X,Y,W):
    Y_pre = predict(X, W)
    acc = (Y_pre == Y).mean()
    return acc

def model(X,Y,alpha,steps,reg):
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    W = np.random.randn(3, X.shape[1]) * 0.0001
    for step in range(steps):
        loss, grad = lossAndGradNaive(X, Y, W, reg)
        W -= alpha*grad
        print("The {} step, loss={}, accuracy={}".format(step,loss,accuracy(X[:,:-1],Y,W)))
    return W

# train_data,train_label,test_data,test_label=load_data(0.2)

train_data = train_set.tdm
train_label = train_set.label
test_data = test_set.tdm
test_label = test_set.label
W = model(train_data, train_label, 0.0001, 25, 0.5)
print("Test accuracy of the model is {}".format(accuracy(test_data, test_label, W)))
