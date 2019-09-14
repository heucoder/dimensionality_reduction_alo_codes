# coding:utf-8
import numpy as np
import sys
sys.path.append("..")

from utils import *
from decoraters import *

def mds(dist, n_dims):
    # dist (n_samples, n_samples)
    dist = dist**2
    n = dist.shape[0]
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1)/n
    T3 = np.sum(dist, axis = 0)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]

    return picked_eig_vector*picked_eig_val**(0.5)
    
@check
def ISOMAP(data,n=2,n_neighbors=30):
    D = cal_pairwise_dist(data)**0.5
    D_floyd=floyd(D, n_neighbors)
    data_n = mds(D_floyd, n_dims=n)
    return data_n

