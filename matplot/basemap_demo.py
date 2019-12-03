from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import netCDF4 as nc

def demo1():
    # 新建地图
    basic_nc_file = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1981.nc"
    plt.figure(figsize=(16, 16), dpi=200, )
    nc_data = nc.Dataset(basic_nc_file)
    for x in nc_data.variables.keys():
        print(x, nc_data.variables[x][:].shape())
    lons = nc_data.variables['lon'][:][0, :, :]
    lats = nc_data.variables['lat'][:][0, :, :]

    lon_0 = lons.mean()
    lat_0 = lats.mean()
    output_data = nc_data.variables['pm25'][:][0, :, :]

    map = Basemap(lat_0=lat_0, lon_0=lon_0, projection='lcc' ) #Basemap类有很多属性，这里全都使用默认参数

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