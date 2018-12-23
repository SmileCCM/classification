#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from sklearn import neighbors
import numpy as np
from sklearn.svm import LinearSVC

from Tools import readbunchobj


def knnsvm_train(x_train, y_train, x_test, y_test, numk, rfile):
    h = .01
    # 14 is the best
    neigh = neighbors.NearestNeighbors(n_neighbors=numk)
    neigh.fit(x_train)
    # print(x)
    all_label = []
    index = 0
    right = 0
    test_result = open(rfile, 'w')
    for one_test in x_test:

        result = neigh.kneighbors(one_test)
        label_index = result[1]
        label = []
        train = []
        # array_label = label_index.copy()
        for i in label_index:
            c = 0
            for j in i:
                one_label = y_train[j]
                one_train = x_train[j]
                label.append(one_label)
                train.append(one_train)
                # array_label[0][c]=one_label
                c = c + 1
        print(len(label), len(train))
        if len(set(label)) == 1:
            test_result.write(str(label[0]) + '\n')
            if label[0] == y_test[index]:
                right = right + 1

        else:
            np_label = np.array(label)
            np_train = np.array(train)
            # np_label = label
            # np_train = train
            print(len(np_label), len(np_train))
            clf = LinearSVC()
            clf.decision_function_shape = 'ovr'
            print(np_train.shape, np_label.shape)
            # print (result[0], array_label)
            clf.fit(np_train, np_label)
            test_result.write(str(clf.predict(one_test)[0]) + '\n')
            if y_test[index] == clf.predict(one_test)[0]:
                right = right + 1

        index = index + 1
    # print right
    print(float(right) / float(len(y_test)))


if __name__ == '__main__':
    # 导入训练集
    trainpath = "train_word_bag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "test_word_bag/testspace.dat"
    test_set = readbunchobj(testpath)


    knnsvm_train(train_set.tdm, train_set.label, test_set.tdm, test_set.label, 125, "result")