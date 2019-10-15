#coding=utf-8
import numpy as np
import sys,os
import netCDF4
from netCDF4 import Dataset
import h5py

def readNC(pth):
    nc_obj = Dataset(pth)
    pm25 = (nc_obj.variables['pm25'][:])
    pm10 = (nc_obj.variables['pm10'][:])
    so2 = (nc_obj.variables['so2'][:])
    no2 = (nc_obj.variables['no2'][:])
    co = (nc_obj.variables['co'][:])
    psfc = (nc_obj.variables['psfc'][:])
    u = (nc_obj.variables['u'][:])
    v = (nc_obj.variables['v'][:])
    temp = (nc_obj.variables['temp'][:])
    rh = (nc_obj.variables['rh'][:])
    res = np.concatenate([pm25, pm10, so2, no2, co, psfc, u, v, temp, rh])
    return np.asarray(res).transpose(1,2,0) #(339,432,10)

def readNpy(path):
    data=np.load(path)
    return data


if __name__=='__main__':
    filepth='/home/th/daqisuo/IAP/Reanalysis/2017010100'
    files=os.listdir(filepth)
    savepth='./daqisuo_hdf5'
    if not os.path.exists(savepth):
        os.makedirs(savepth)
    for f in files:
        h5file = h5py.File(os.path.join(savepth,f[:-3]+'.h5'), 'w')
        res=readNC(os.path.join(filepth,f))
        h5file.create_dataset(f[:-3], data=res)
        h5file.close()   
        #np.save(os.path.join(savepth,f[:-3]),res)
        
    #npypath="/home/th/daqisuo/IAP/Reanalysis/npy/CN-Reanalysis2017123023.npy"
    #data=readNpy(npypath)
    #print(data.shape)
