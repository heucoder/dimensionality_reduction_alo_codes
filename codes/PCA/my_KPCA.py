# coding:utf-8
# 实现KPCA

from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
import numpy as np

from scipy.spatial.distance import pdist, squareform


def sigmoid(x, coef = 0.25):
    x = np.dot(x, x.T)
    return np.tanh(coef*x+1)

def linear(x):
    x = np.dot(x, x.T)
    return x

def rbf(x, gamma = 15):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma*mat_sq_dists)

# data (n_samples, n_features)
# n_components target dim
# kernel kernel fun
def my_KPCA(data, n_components=2, kernel = rbf):

    K = kernel(data)
    #
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    #
    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1]
    eigval = eig_values[idx][:n_components]
    eigvector = eig_vector[:, idx][:, :n_components]
    print(eigval)
    eigval = eigval**(1/2)
    vi = eigvector/eigval.reshape(-1,n_components)
    data_n = np.dot(K, vi)
    return data_n


if __name__ == "__main__":
    data = load_iris().data
    target = load_iris().target
    data_1 = my_KPCA(data, kernel=sigmoid)

    print("------------")

    kpca = KernelPCA(n_components=2, kernel="sigmoid")
    data_2 = kpca.fit_transform(data)
    print(kpca.lambdas_)
