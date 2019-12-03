from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

def demo1():
    # 新建地图
    basic_nc_file = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1981.nc"
    plt.figure(figsize=(16, 16), dpi=200, )
    nc_data = nc.Dataset(basic_nc_file)
    keys =  nc_data.variables.keys()
    for x in keys:
        tmp = np.array(nc_data.variables[x][:])
        print(tmp.shape)
        print(x, )
    lons = nc_data.variables['lon'][:]
    lats = nc_data.variables['lat'][:]
    print(lons)
    print(lats)

    lon_0 = lons.mean()
    lat_0 = lats.mean()
    output_data = nc_data.variables['tmax'][:][0]

    map = Basemap(lat_0=lat_0, lon_0=lon_0, projection='lcc', llcrnrlon=50,urcrnrlon=180,
             llcrnrlat=-20,urcrnrlat=65) #Basemap类有很多属性，这里全都使用默认参数

    # 画图
    map.drawcoastlines()

    # 显示结果
    plt.show()

    # 存储结果
    plt.savefig('./test.png')

def unit_test():
    demo1()

if __name__ == '__main__':
    unit_test()