# coding:utf-8
import numpy as np
import sys
sys.path.append("..")

from utils import *
from decoraters import *
'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''
@check
def LLE(data, n_dims = 2, n_neighbors = 10):
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

    print("Xi.shape", Xi.shape)
    print("Ni.shape", Ni.shape)
    print("Si.shape", Si.shape)

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
