#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Tools import readbunchobj
stat = time.time()
# 导入训练集
trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
'''
def kNN(X_train, y_train, X_test, y_test):
    total_corr = 0
    i = 0
    for clas in y_test:
        # for every image in test set compute
        predict = X_test[i]
        distances = []
        j = 0
        for group in y_train:
            # Calculate the euclidean distance between test image with every train image
            features = X_train[j]
            euclidean_distance = np.linalg.norm(np.subtract(features, predict))
            distances.append([euclidean_distance, group])
            j += 1

        # Sorte the distance in ascending order and take the 1st one as the result
        result = [k[1] for k in sorted(distances)[:1]]
        predicted = result[0]
        correct = clas

        # Compare the result with original class and computed the accuracy
        c = np.sum(predicted == correct)
        total_corr += c
        i += 1
    print('Accuracy is ', float(total_corr)/80*100)
    return float(total_corr)/80*100
# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
kNN(train_set.tdm, train_set.label,test_set.tdm, test_set.label)
'''
clf = KNeighborsClassifier(n_neighbors=125).fit(train_set.tdm, train_set.label)
# 预测分类结果
predicted = clf.predict(test_set.tdm)
'''
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
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

