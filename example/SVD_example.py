
from sklearn.datasets import load_iris
import sys
sys.path.append("..")

from feature_extract import SVD

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    U, Sigma, VT = SVD.SVD(X)