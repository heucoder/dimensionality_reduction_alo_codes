#coding:utf-8

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
import sys
sys.path.append("..")

from feature_extract import KPCA

if __name__ == "__main__":
    data = load_iris().data
    Y = load_iris().target
    data_1 = KPCA.KPCA(data, kernel='rbf')


    sklearn_kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
    data_2 = sklearn_kpca.fit_transform(data)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_KPCA")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_KPCA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.show()
