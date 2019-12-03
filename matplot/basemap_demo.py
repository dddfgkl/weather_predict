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
    # 新建地图
    basic_nc_file = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1989.nc"
    single_cpc_nc_prec_path = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1982.nc"

    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print(raw_data.shape)

    plt.figure(figsize=(16, 16), dpi=200, )
    nc_data = nc.Dataset(single_cpc_nc_prec_path)
    keys =  nc_data.variables.keys()
    # for x in keys:
    #     tmp = np.array(nc_data.variables[x][:])
    #     print(x, tmp.shape)
    #     print()
    lons = nc_data.variables['lon'][:].data
    lats = nc_data.variables['lat'][:].data

    zuo = lons[0]
    you = lons[-1]
    shang = lats[0]
    xia = lats[-1]
    print("#### basic info ###", zuo, you, shang, xia)

    n_lons = []
    n_lats = lats
    n_lons = [lons for i in range(len(lats))]
    # n_lats = [lats for i in range(len(lons))]
    # n_lats = n_lons
    lons = np.array(n_lons)
    lats = np.zeros(lons.shape)
    for i in range(54):
        for j in range(87):
            lats[i][j] = n_lats[i]


    print(lons.shape, type(lons))
    print(lats.shape, type(lats))


    lon_0 = lons.mean()
    lat_0 = lats.mean()

    # output_data = nc_data.variables['tmax'][:].data
    output_data = raw_data[0][0]
    # print("output shape", output_data)
    # for x in range(len(output_data)):
    #     for y in range(len(output_data[0])):
    #         # print(type(output_data[x][y]))
    #         if np.isnan(output_data[x][y]) or output_data[x][y] < -100:
    #             # print("get !!!!!")
    #             output_data[x][y] = 0.0

    m = Basemap(lat_0=lat_0, lon_0=lon_0, projection='lcc', llcrnrlon=zuo,urcrnrlon=you,
             llcrnrlat=shang,urcrnrlat=xia) #Basemap类有很多属性，这里全都使用默认参数

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