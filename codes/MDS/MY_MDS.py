import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def cal_pairwise_dist(x):
    # '''计算pairwise 距离, x是matrix
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    # '''
    sum_x = np.sum(np.square(x), 1)
    # print -2 * np.dot(x, x.T)
    # print np.add(-2 * np.dot(x, x.T), sum_x).T
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离
    return dist**0.5


def my_mds(data, n_dims):
    # data (n_samples, n_features)
    n, d = data.shape
    dist = cal_pairwise_dist(data)
    dist = dist**2
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1, keepdims=True)/n
    T3 = np.sum(dist, axis = 0, keepdims=True)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    # print(picked_eig_vector.shape, picked_eig_val.shape)
    return picked_eig_vector*picked_eig_val**(0.5)


if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    y = iris.target
    d = my_mds(data, 2)
    print(d.shape)
    plt.scatter(d[:, 0], d[:, 1], c = y)
    plt.show()