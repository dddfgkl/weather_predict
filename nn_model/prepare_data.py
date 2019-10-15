#coding=utf-8
import readgrds.grds as grds
import h5py
import os
from collections import defaultdict


ctl_pth = "../naqpd02ctl"
allctl = os.listdir(ctl_pth)
allctl.sort()
#print("allctl: ",allctl,len(allctl))

grd_pth = "../naqpd02grd"
ctl_grd_dict = defaultdict(list)

grd_dir = os.listdir(grd_pth)

for ctl in allctl:
    for grd in grd_dir:
        if ctl.split('.')[1]==grd:
            grdfiles=os.listdir(os.path.join(grd_pth,grd))
            for f in grdfiles:
                filepath = os.path.join(grd_pth,grd,f)
                ctl_grd_dict[ctl].append(filepath)
            
#print(ctl_grd_dict[allctl[0]])     

vars=['U','V','T','P','HGT','RAIN','PBL','RH','SO2','NO2','CO','O3','ASO4','ANO3','ANH4','BC','OC','PPMF','PPMC','SOA','TPM25','TPM10','O3_8H','VISIB','AOD','EXT']

for ctl in allctl:
    if ctl in ctl_grd_dict:
        var_dir = os.path.join(os.getcwd(), 'data', ctl.split('.')[1])
        if not os.path.exists(var_dir):
            os.makedirs(var_dir)
        for v in vars:
            h5file = h5py.File(os.path.join(var_dir, v+'.h5'), 'w')
            for grd in ctl_grd_dict[ctl]:
                file = grds.Grds(os.path.join(ctl_pth, ctl), grd)
                var = file.read(v) # 返回的数据，为三维的numpy格式（z, y, x）
                h5file.create_dataset(os.path.basename(grd)[:-4]+v, data=var)
            h5file.close()
