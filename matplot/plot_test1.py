from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import h5py
import os

def read_h5(file_path, key):
    if os.path.exists(file_path) == False:
        print("file not exist")
        raise FileExistsError
    f = h5py.File(file_path, 'r')
    return f[key][:]

def demo1():

    n_lons = [[60,70],[80,90]]
    n_lats = [[0,10],[20,30]]
    # n_lats = n_lons
    lons = np.array(n_lons)
    lats = np.array(n_lats)
    # lats = lats.transpose((1, 0))


    print(lons.shape, type(lons))
    print(lats.shape, type(lats))


    lon_0 = lons.mean()
    lat_0 = lats.mean()

    # output_data = nc_data.variables['tmax'][:].data
    output_data = [[1,2],[3,4]]
    output_data = np.array(output_data)
    # print("output shape", output_data)
    # for x in range(len(output_data)):
    #     for y in range(len(output_data[0])):
    #         # print(type(output_data[x][y]))
    #         if np.isnan(output_data[x][y]) or output_data[x][y] < -100:
    #             # print("get !!!!!")
    #             output_data[x][y] = 0.0

    m = Basemap(lat_0=lat_0, lon_0=lon_0, projection='lcc', llcrnrlon=50,urcrnrlon=180,
             llcrnrlat=-20,urcrnrlat=65) #Basemap类有很多属性，这里全都使用默认参数

    # 画图
    m.drawcoastlines()
    # m.drawparallels(np.arange(16., 54., 10.), labels=[1, 0, 0, 0], fontsize=20)
    # m.drawmeridians(np.arange(75., 135., 10.), labels=[0, 0, 0, 1], fontsize=20)
    xi, yi = m(lons, lats)
    print(xi.shape, yi.shape)
    print(output_data.shape)
    cs = m.contourf(xi, yi, output_data, cmap=plt.cm.get_cmap('jet'))
    cbar = m.colorbar(cs, location='right', pad='5%', )
    cbar.set_label('μg/m$^3$', fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    plt.title('PM25', fontsize=25)
    plt.xticks(fontsize=24, )
    plt.yticks(fontsize=24, )
    plt.savefig('pm25')

    # 显示结果
    # plt.show()

    # 存储结果
    plt.savefig('./test.png')

def unit_test():
    demo1()

if __name__ == '__main__':
    unit_test()