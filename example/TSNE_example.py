# coding:utf-8

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from feature_extract import TSNE

if __name__ == "__main__":
    digits = load_digits()
    X = digits.data
    Y = digits.target

    data_2d = TSNE.TSNE(X, 2, 30, 100)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c = Y)
    plt.show()