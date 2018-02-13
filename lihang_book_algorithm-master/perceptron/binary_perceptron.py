# encoding=utf-8
# @Author: WenDesi
# @Date:   09-08-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

'''一层的感知器模型，用于进行0和非0的两类分类'''

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        # # 将w的每一位和x的图像的每一个像素点相乘并求和，如果和大于0，则返回1，对应非零数字，如果和小于0，返回0，对应数字0
        wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)

        correct_count = 0
        time = 0

        while time < self.max_iteration:
            # 随机取一个索引
            index = random.randint(0, len(labels) - 1)
            # 取索引为index的图像的数据
            x = list(features[index])
            x.append(1.0)
            # label 0:-1;  label 1:1
            y = 2 * labels[index] - 1
            # 将w的每一位和x的图像的每一个像素点相乘并求和
            wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])

            # 1）对于数字0，有像素值的点对应的w权值为负数，或者由于其他数字的干扰，有可能部分点的w值为正，但毕竟是少数，因此其乘积总和为负数；
            # 2）其他点为0，对应w权值为0或者正数，或者也有可能其他不同写法的数字0导致该点w值为负，但对当前图像，该像素点值为0，
            # 因此1）+2）其乘积累加为0，因此总和应该为负数，
            # 3）对于非零数字，有像素值的点对应的权值为正，即使有少部分点因为数字0的影响其对应的w值为负，但毕竟是少数，因此其乘积总和应该为正数；
            # 4）其他点为0，对应w权值为0或者负数，或者有些由于其他数字影响权值w是正的，但当前图像该像素点为0，因此其乘积累加为0，
            # 因此3）+4）总和应该为正数；
            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
            #
            # 迭代w，对于数字0的图像上的每一点的灰度值，乘以y=-1，其值为负数
            # 对于数字非零的图像上的每一点的灰度值，乘以y=1，其值为正数
            for i in xrange(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print 'Start read data'

    time_1 = time.time()

    #读取图像数据，每个图像784位，数字0的标签为0，非0的标签为1
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'

    print 'Start training'
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'Start predicting'
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre is ", score
