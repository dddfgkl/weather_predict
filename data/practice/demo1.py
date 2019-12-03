#encoding=utf-8
# @Author:MC
# @Last Edit: 2019.10.17
# @Desc:

import numpy as np

def demo1():
    a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]]
    a = np.array(a)
    print(a.shape)
    a = a.transpose(2,1,0)
    print(a, a.shape)
    a = a.transpose(2,1,0)
    print(a, a.shape)
    print()

def demo2():
    a = np.array([1,2,3,4,5,6])
    b = a.reshape(2,3)
    print(b)

if __name__ == '__main__':
    demo2()
