#coding:utf-8
from functools import wraps
import numpy as np

def check(func):
    @wraps(func)
    def wrapper(*argc, **kw):
        data = argc[0]
        try:
            data = np.array(data).astype(np.float64)
        except Exception:
            raise("error: data's shape or type is wrong!")
        
        if (len(argc) > 1):
            assert isinstance(argc[1], int)
        
        return func(data, *argc[1:], **kw)

    return wrapper

@check
def f(data, dim, c = 2, s = 5):
    print(data, dim) 

if __name__ == "__main__":
    # a = [["1111"], [2,3,4]]
    b = np.array([[1,2,3], [4,5,6]])
    c = np.mat([[1,2,3], [4,5,6]])

    # f(a,2)
    f(b,2)
    f(c,2)