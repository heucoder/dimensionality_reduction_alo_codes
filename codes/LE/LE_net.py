from numpy import *
import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# code from net!!!
# code from net!!!
# code from net!!!


def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    #Generate a swiss roll dataset.
    t = 1.5 * np.pi * (1 + 2 * random.rand(1, n_samples))
    x = t * np.cos(t)
    y = 83 * random.rand(1, n_samples)
    z = t * np.sin(t)
    X = np.concatenate((x, y, z))
    X += noise * random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    return X, t

def laplaEigen(dataMat,k,t):
    m,n=shape(dataMat)
    W=mat(zeros([m,m]))
    D=mat(zeros([m,m]))
    for i in range(m):
        k_index=knn(dataMat[i,:],dataMat,k)
        for j in range(k):
            sqDiffVector = dataMat[i,:]-dataMat[k_index[j],:]
            sqDiffVector=array(sqDiffVector)**2
            sqDistances = sqDiffVector.sum()
            W[i,k_index[j]]=math.exp(-sqDistances/t)
            D[i,i]+=W[i,k_index[j]]
    L=D-W
    print(D[0,0])
    Dinv=np.linalg.inv(D)
    X=np.dot(Dinv,L)
    lamda,f=np.linalg.eig(X)
    return lamda, f, D

def knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    return sortedDistIndicies[0:k]



dataMat, color = make_swiss_roll(n_samples = 2000)


lamda, f, D = laplaEigen(dataMat, 15, 5)
fm,fn =shape(f)
print('fm,fn:',fm,fn)
lamdaIndicies = argsort(lamda)

first=0
second=0
print(lamdaIndicies[0], lamdaIndicies[1])

for i in range(fm):
    if lamda[lamdaIndicies[i]].real>1e-5:
        print(lamda[lamdaIndicies[i]])
        first=lamdaIndicies[i]
        second=lamdaIndicies[i+1]
        break

print("first, second: ", first, second)
redEigVects = f[:,lamdaIndicies]
print("lamda: ", lamda[first], lamda[second])
#
# plt.scatter(dataMat[:, 0])
#
# plt.scatter(array(redEigVects[:, first]).flatten(), array(redEigVects[:, second]).flatten(), c = color)
# plt.show()

data_ndim = array(f[:,[first, second]])
print(np.dot(np.dot(data_ndim.T, D), data_ndim))

fig=plt.figure('origin')
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(dataMat[:, 0], dataMat[:, 1], dataMat[:, 2], c=color,cmap=plt.cm.Spectral)
fig=plt.figure('lowdata')
ax2 = fig.add_subplot(111)

ax2.scatter(array(f[:,first]).flatten(), array(f[:,second]).flatten(), c=color, cmap=plt.cm.Spectral)
plt.show()