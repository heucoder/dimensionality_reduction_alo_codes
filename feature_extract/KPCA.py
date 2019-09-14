# coding:utf-8

import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys
sys.path.append('..')
'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''

from decoraters import *

def pair_sigmoid(x, coef = 0.25):
    x = np.dot(x, x.T)
    return np.tanh(coef*x+1)

def pair_linear(x):
    x = np.dot(x, x.T)
    return x

def pair_rbf(x, gamma = 15):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma*mat_sq_dists)

kernel_dict={'rbf': pair_rbf, 'linear': pair_linear, 'sigmoid': pair_sigmoid}

@check
def KPCA(data, n_dims=2, kernel = 'rbf'):
    '''

    :param data: (n_samples, n_features)
    :param n_dims: target n_dims
    :param kernel: kernel functions
    :return: (n_samples, n_dims)
    '''

    kernel = kernel_dict[kernel]

    K = kernel(data)
    #
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    #
    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1]
    eigval = eig_values[idx][:n_dims]
    eigvector = eig_vector[:, idx][:, :n_dims]
    print(eigval)
    eigval = eigval**(1/2)
    vi = eigvector/eigval.reshape(-1,n_dims)
    data_n = np.dot(K, vi)
    return data_n
