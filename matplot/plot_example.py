'''
hbwang 2019.6.20
case5
'''
import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import datetime

plt.switch_backend('agg')  # do for avoid the wrong information of RuntimeError: Invalid DISPLAY variable
print('start','',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

data = nc.Dataset('CN-Reanalysis2017010100.nc')
keys =  data.variables.keys()
for x in keys:
    tmp = np.array(data.variables[x][:])
    print(tmp.shape)
    print(x)

lons = data.variables['lon2d'][:].data[0,:,:]
lats = data.variables['lat2d'][:].data[0,:,:]

print("########", lons.shape, type(lons))
print("########", lats.shape,type(lats))
lon_0 = lons.mean()
lat_0 = lats.mean()
output_data = data.variables['pm25'][:][0,:,:]

plt.figure(figsize=(16,16),dpi=200,)
m = Basemap(lat_0=lat_0,lon_0=lon_0,projection='lcc',
             llcrnrlon=80,urcrnrlon=140,
             llcrnrlat=12,urcrnrlat=54,)
m.drawcountries()
m.drawparallels(np.arange(16., 54., 10.), labels=[1,0,0,0], fontsize=20)
m.drawmeridians(np.arange(75., 135., 10.), labels=[0,0,0,1], fontsize=20)
xi,yi = m(lons,lats)
print(xi.shape, yi.shape)
print(output_data.shape)
cs = m.contourf(xi,yi,output_data,cmap=plt.cm.get_cmap('jet'))
cbar = m.colorbar(cs,location='right',pad='5%',)
cbar.set_label('μg/m$^3$',fontsize=22)
cbar.ax.tick_params(labelsize=24)
plt.title('PM25',fontsize=25)
plt.xticks(fontsize=24,)
plt.yticks(fontsize=24,)
# plt.show()
plt.savefig('./test1.png')




