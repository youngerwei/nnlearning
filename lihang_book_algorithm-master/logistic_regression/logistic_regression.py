# encoding=utf-8
# @Author: WenDesi
# @Date:   08-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

'''
该例子只用于判断1和0两个数字。所有训练数据都是1和0的图像数据。
通过w参数特征向量来判断图像属于哪个数字，w的长度和图像像素点数量相同
1、初始化特征向量数组w为0；
2、随机选取1个label，计算w向量和图像每个像素点的点积wx，然后以wx为参数通过logistic公式计算结果是1还是0；
3、如果结果和图像label一致，进行正确判断的累加，累加到一定数量停止循环；如果不一致，则通过梯度下降法迭代特征向量w；
4、迭代方法为 w(i)=迭代步长×x(i)×logistic函数。

logistic回归模型的原理：是一种2类分类器，根据定义的logistic公式，使用训练数据来估计公式中的w，使所有样本都符合2类分类器的分类结果要求。
然后用求得的w来估计未知输入，确定其分类。
计算w时采用最大似然法，给出P(y=1|x)=π及P(y=0|x)=1-π的最大似然估计函数，然后计算其极值。极值计算可以采用梯度下降法或拟牛顿法。
'''

import time
import math
import random

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])
        exp_wx = math.exp(wx)

        predict1 = exp_wx / (1 + exp_wx)
        predict0 = 1 / (1 + exp_wx)

        if predict1 > predict0:
            return 1
        else:
            return 0

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)

        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = labels[index]

            if y == self.predict_(x):
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            # print 'iterater times %d' % time
            time += 1
            correct_count = 0

            # calculate the parameter w using 最大似然估计法，通过计算对数形式的最大似然函数的极值计算w，
            # 对数形式的最大似然函数的极值通过随机梯度下降法求得

            # 先计算logistic公式中的参数和输入的点乘(w·x)和exp(w·x)
            wx = sum([self.w[i] * x[i] for i in xrange(len(self.w))])
            exp_wx = math.exp(wx)

            # 用最大似然估计法计算对数形式的最大似然函数取极值是w的数值。
            # 随机梯度下降法计算梯度，损失函数为-L, L为对数形式的最大似然函数。
            for i in xrange(len(self.w)):
                self.w[i] -= self.learning_step * \
                             (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))

    def predict(self, features):
        labels = []

        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))

        return labels


if __name__ == "__main__":
    print 'Start read data'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)

    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'

    print 'Start training'
    lr = LogisticRegression()
    lr.train(train_features, train_labels)

    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'Start predicting'
    test_predict = lr.predict(test_features)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre is ", score
