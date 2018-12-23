#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from Tools import readbunchobj

def PCA(X_train, X_test, k):
    # Centerize the images将图像集中
    X_train -= np.mean(X_train, axis=0)
    X_test -= np.mean(X_test, axis=0)

    print('Calculating Covariance matrix')  # 计算协方差矩阵
    CovM = np.cov(X_train.T)
    # 计算的特征值和特征向量
    print('Calculating eigen values and eigen vectors, please wait...')
    evals, evecs = np.linalg.eigh(CovM)
    # Sort the eigen values in descending order and then sorted the eigen vectors by the same index
    # 按降序对特征值进行排序，然后按照相同的索引对特征向量进行排序
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]

    # Can uncomment for plotting eigen values graph
    # evals = evals[idx]
    # pyplot.plot(evals)
    # pyplot.show()
    return np.dot(evecs.T, X_train.T).T, np.dot(evecs.T, X_test.T).T

def LDA(X_train, y_train, X_test, k):

    print('Calculating class wise mean vectors')
    m, n = X_train.shape
    class_wise_mean = []
    for i in range(1, 41):
        idx = np.where(y_train == i)
        class_wise_mean.append(np.mean(X_train[idx], axis=0))

    print('Calculating within-class scatter matrix')
    within_SM = np.zeros((n, n))
    for i, mean_vector in zip(range(1, 41), class_wise_mean):
        class_wise_M = np.zeros((n, n))
        idx = np.where(y_train==i)
        for img in X_train[idx]:
            img, mean_vector = img.reshape(n, 1), mean_vector.reshape(n, 1)
            class_wise_M += (img - mean_vector).dot((img - mean_vector).T)
        within_SM += class_wise_M

    print('Calculating between-class scatter matrix')
    total_mean = np.mean(X_train, axis=0)
    between_SM = np.zeros((n, n))
    for i, mean_vector in enumerate(class_wise_mean):
        idx = np.where(y_train==i+1)
        cnt = X_train[idx].shape[0]
        mean_vector = mean_vector.reshape(n, 1)
        total_mean = total_mean.reshape(n, 1)
        between_SM += cnt * (mean_vector - total_mean).dot((mean_vector - total_mean).T)

    print('Calculating eigen values and eigen vectors, please wait...')
    evals, evecs = np.linalg.eigh(np.linalg.inv(within_SM).dot(between_SM))
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]

    # Can uncomment for plotting eigen values graph
    # evals = evals[idx]
    # pyplot.plot(evals)
    # pyplot.show()
    return np.dot(evecs.T, X_train.T).T, np.dot(evecs.T, X_test.T).T

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

if __name__ == "__main__":
    print('\nRunning for PCA + LDA + 1NN')

    # 导入训练集
    trainpath = "train_word_bag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "test_word_bag/testspace.dat"
    test_set = readbunchobj(testpath)
    X_train = train_set.tdm
    y_train = train_set.label
    X_test = test_set.tdm
    y_test = test_set.label
    # X_train_pca, X_test_pca = PCA(X_train, X_test, 70)
    # X_train_pca_lda, X_test_pca_lda = LDA(X_train_pca, y_train, X_test_pca, 25)
    kNN(X_train, y_train, X_test, y_test)