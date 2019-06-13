# coding:utf-8
import numpy as np
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D

'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def get_n_neighbors(data, n_neighbors = 10):
    '''

    :param data: (n_samples, n_features)
    :param n_neighbors: n nearest neighbors
    :return: neighbors indexs
    '''

    dist = cal_pairwise_dist(data)**0.5
    n = dist.shape[0]
    N = np.zeros((n, n_neighbors))

    for i in range(n):
        index_ = np.argsort(dist[i])[1:n_neighbors+1]
        N[i] = N[i] + index_

    return N.astype(np.int32)

def lle(data, n_dims = 2, n_neighbors = 10):
    '''

    :param data:(n_samples, n_features)
    :param n_dims: target n_dims
    :param n_neighbors: n nearest neighbors
    :return: (n_samples, n_dims)
    '''
    N = get_n_neighbors(data, n_neighbors)
    n, D = data.shape

    # prevent Si to small
    if n_neighbors > D:
        tol = 1e-3
    else:
        tol = 0

    # calculate W
    W = np.zeros((n_neighbors, n))
    I = np.ones((n_neighbors, 1))
    for i in range(n):
        Xi = np.tile(data[i], (n_neighbors, 1)).T
        Ni = data[N[i]].T

        Si = np.dot((Xi-Ni).T, (Xi-Ni))
        # magic and why????
        Si = Si+np.eye(n_neighbors)*tol*np.trace(Si)

        Si_inv = np.linalg.pinv(Si)
        wi = (np.dot(Si_inv, I))/(np.dot(np.dot(I.T, Si_inv), I)[0,0])
        W[:, i] = wi[:,0]

    W_y = np.zeros((n, n))
    for i in range(n):
        index = N[i]
        for j in range(n_neighbors):
            W_y[index[j],i] = W[j,i]

    I_y = np.eye(n)
    M = np.dot((I_y - W_y), (I_y - W_y).T)

    eig_val, eig_vector = np.linalg.eig(M)
    index_ = np.argsort(np.abs(eig_val))[1:n_dims+1]
    print("index_", index_)
    Y = eig_vector[:, index_]
    return Y

if __name__ == '__main__':
    X, Y = make_s_curve(n_samples = 500,
                           noise = 0.1,
                           random_state = 42)

    data_1 =lle(X, n_neighbors = 30)

    data_2 = LocallyLinearEmbedding(n_components=2, n_neighbors=30).fit_transform(X)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_LLE")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LLE")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.savefig("LLE.png")
    plt.show()
