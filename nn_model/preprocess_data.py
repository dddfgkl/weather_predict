#coding=utf-8
import numpy as np
import h5py
import os
import time 
from datetime import datetime

data_pth = './data'
data_dir = os.listdir(data_pth)
data_dir.sort()

for d in data_dir:
    dirpth = os.path.join(data_pth, d)
    h5file_list = os.listdir(dirpth)
    res = np.zeros((169,269,239,26))
    for f in h5file_list:
        if f[:-3]=="TPM25":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,0] = tmp
        elif f[:-3]=="SO2":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,1] = tmp
        elif f[:-3]=="NO2":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,2] = tmp
        elif f[:-3]=="CO":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,3] = tmp
        elif f[:-3]=="O3":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,4] = tmp
        elif f[:-3]=="ASO4":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,5] = tmp
        elif f[:-3]=="ANO3":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,6] = tmp
        elif f[:-3]=="ANH4":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,7] = tmp
        elif f[:-3]=="BC":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,8] = tmp
        elif f[:-3]=="OC":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,9] = tmp
        elif f[:-3]=="PPMF":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,10] = tmp
        elif f[:-3]=="PPMC":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,11] = tmp
        elif f[:-3]=="SOA":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,12] = tmp
        elif f[:-3]=="TPM10":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,13] = tmp
        elif f[:-3]=="O3_8H":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,14] = tmp

        elif f[:-3]=="U":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,15] = tmp
        elif f[:-3]=="V":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,16] = tmp
        elif f[:-3]=="T":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,17] = tmp
        elif f[:-3]=="P":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,18] = tmp
        elif f[:-3]=="HGT":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,19] = tmp
        elif f[:-3]=="RAIN":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,20] = tmp
        elif f[:-3]=="PBL":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,21] = tmp
        elif f[:-3]=="RH":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,22] = tmp
        elif f[:-3]=="VISIB":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,23] = tmp
        elif f[:-3]=="AOD":
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,24] = tmp
        else:  #"EXT"
            readFile = h5py.File(os.path.join(dirpth,f),'r')
            key_list= []
            for key in readFile.keys():
                key_list.append(key)
            key_list.sort()
            tmp = np.zeros((169, 269, 239))
            for idx,k in enumerate(key_list):
                tmp[idx]=readFile[k][0]
            res[:,:,:,25] = tmp
    h5file = h5py.File(os.path.join('./pre_data',d+'.h5'), 'w')
    h5file.create_dataset(d, data=res)
    h5file.close()
