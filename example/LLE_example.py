#coding:utf-8
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("..")

from feature_extract import LLE

if __name__ == '__main__':
    X, Y = make_s_curve(n_samples = 500,
                           noise = 0.1,
                           random_state = 42)

    data_1 =LLE.LLE(X, n_neighbors = 30)

    data_2 = LocallyLinearEmbedding(n_components=2, n_neighbors=30).fit_transform(X)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_LLE")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LLE")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    # plt.savefig("LLE.png")
    plt.show()
