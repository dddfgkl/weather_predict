import torch
import h5py
import os
from netCDF4 import Dataset

def demo1():
    a = [1,2,3,4]
    x1, x2, x3x4 = a
    print(x1)

def demo2():
    a = [100, 200, 300]
    for i, x in enumerate(a):
        print(i, x)

def demo3():
    for i in range(2, 100):
        print(i)

if __name__ == '__main__':
    demo3()