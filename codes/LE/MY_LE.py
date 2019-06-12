# coding:utf-8

import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    #Generate a swiss roll dataset.
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = 83 * np.random.rand(1, n_samples)
    z = t * np.sin(t)
    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    return X, t

def rbf(dist, t = 1.0):
    return np.exp(-(dist/t))

def cal_pairwise_dist(x):
    # x (n_samples, n_features)
    # '''计算pairwise 距离, x是matrix
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    # '''
    sum_x = np.sum(np.square(x), 1)
    # print -2 * np.dot(x, x.T)
    # print np.add(-2 * np.dot(x, x.T), sum_x).T
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离
    return dist**0.5

def cal_rbf_dist(data, n_neighbors = 10, t = 1):
    # dist (n_samples, n_samples)
    dist = cal_pairwise_dist(data)**2
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:n_neighbors+1]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W

def MY_LE(data,
          n_dims = 2,
          n_neighbors = 10, t = 1.0):

    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)
    for i in range(N):
        D[i,i] = np.sum(W[i])

    D_inv = np.linalg.inv(D)
    L = D - W
    eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))

    sort_index_ = np.argsort(eig_val)

    eig_val = eig_val[sort_index_]
    j = 0
    for v in eig_val:
        j+=1
        if v > 1e-5:
            break
    print("j", j)

    sort_index_ = sort_index_[j:j+n_dims]
    print(sort_index_)
    eig_val_picked = eig_val[sort_index_]
    eig_vec_picked = eig_vec[:, sort_index_]
    print(eig_val_picked)
    print(np.dot(np.dot(eig_vec_picked.T, D), eig_vec_picked))

    X_ndim = eig_vec_picked
    return X_ndim

if __name__ == '__main__':
    X, Y = make_swiss_roll(n_samples = 2000)
    # X[:, 1] = X[:, 1]*10
    X_ndim = MY_LE(X, n_neighbors = 20, t = 10)
    print(X_ndim[:,0].shape)

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c = Y)

    ax2 = fig.add_subplot(122)
    ax2.scatter(X_ndim[:, 0], X_ndim[:, 1], c = Y)
    plt.show()