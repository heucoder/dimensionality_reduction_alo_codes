# coding:utf-8
import numpy as np
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

def floyd(D,n_neighbors=15):
    Max = np.max(D)*1000
    n1,n2 = D.shape
    k = n_neighbors
    D1 = np.ones((n1,n1))*Max
    D_arg = np.argsort(D,axis=1)
    for i in range(n1):
        D1[i,D_arg[i,0:k+1]] = D[i,D_arg[i,0:k+1]]
    for k in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i,k]+D1[k,j]<D1[i,j]:
                    D1[i,j] = D1[i,k]+D1[k,j]
    return D1

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def my_mds(dist, n_dims):
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

def my_Isomap(data,n=2,n_neighbors=30):
    D = cal_pairwise_dist(data)**0.5
    D_floyd=floyd(D, n_neighbors)
    data_n = my_mds(D_floyd, n_dims=n)
    return data_n

def scatter_3d(X, y):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    plt.show()

if __name__ == '__main__':
    X, Y = make_s_curve(n_samples = 500,
                           noise = 0.1,
                           random_state = 42)

    data_1 = my_Isomap(X, 2, 10)

    data_2 = Isomap(n_neighbors = 10, n_components = 2).fit_transform(X)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_Isomap")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_Isomap")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.savefig("Isomap.png")
    plt.show()