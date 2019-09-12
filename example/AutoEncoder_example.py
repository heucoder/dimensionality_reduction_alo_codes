# coding:utf-8

from sklearn.datasets import load_iris, load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

sys.path.append('..')

from feature_extract import AutoEncoder

'''
author: heucoder
email: 812860165@qq.com
date: 2019.9.12
'''

if __name__ == '__main__':
    iris = load_digits()
    X = iris.data
    Y = iris.target
    data_1 = AutoEncoder.AutoEncoder(X, [2], learning_rate = 0.2,  n_epochs = 1000)

    data_2 = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_AutoEncoder")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    # plt.savefig("AutoEncoder.png")
    plt.show()