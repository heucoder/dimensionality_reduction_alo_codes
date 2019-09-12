#coding:utf-8
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from feature_extract import LDA

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    data_1 = LDA.LDA(X, Y, 2)

    data_2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, Y)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_LDA")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    # plt.savefig("LDA.png")
    plt.show()