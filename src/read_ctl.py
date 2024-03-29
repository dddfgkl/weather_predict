import numpy as np
import h5py
import csv
from collections import namedtuple

fileName = "/home/machong/PM25-work/Hind3_daily/predict_data/IAP41_Hindcast_SEasian_daily_Tmax_ens_mean_87x54x180x32.bin"
fileName_windows = "../../data/IAP41_Hindcast_SEasian_daily_Tmax_ens_mean_87x54x180x32.bin"
ens_mean_file = "/home/machong/PM25-work/Hind3_daily/ens_mean.ctl"
ens_mean_file_windows = "../../data/ens_mean.ctl"

def read_from_ctl(fileName):
    if fileName is None:
        raise FileExistsError
    a = np.fromfile(fileName, dtype=float)
    print(a.shape, type(a))
    # a_reshape = np.reshape(a, (32, 87, 54, 180))
    # print(a_reshape.shape)

''' grd文件接口 '''
class ControlFile:
    '''
    解析grd的控制文件
    '''
    def __init__(self, ctlFileName):
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.nt = 0
        self.vars = {}
        self.parser(ctlFileName)

    def parser(self, ctlFile):
        ''' 读取配置文件 '''
        infoDic, varsLst = {}, []
        with open(ctlFile, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            cols = line.split()
            # print(line, "  ", cols)
            if line[:1].isalpha() and len(cols) > 1:
                keyName = cols[0].upper()
                varsLst.append(keyName)
                infoDic[keyName] = cols[1:]

        self.get_mete(infoDic, varsLst)
        print(self.nx, self.ny, self.nt, self.nz, self.vars)
        print(varsLst)
        print(infoDic)

    def get_mete(self, infoDic, varsLst):
        ''' 解析配置文件的信息 '''
        if  "PDEF" in infoDic.keys():
            self.nx = int(infoDic["PDEF"][0])
            self.ny = int(infoDic["PDEF"][1])
        else:
            self.nx = int(infoDic["XDEF"][0])
            self.ny = int(infoDic["YDEF"][0])
            self.nt = int(infoDic["TDEF"][0])
            self.nz = int(infoDic["ZDEF"][0])

        recods = namedtuple("recods", "rec  nz")
        rec = (0, 0) # conventionally the end of rec is not used
        for var in varsLst[varsLst.index("VARS")+1:]:
            nz = int(infoDic[var][0])
            nz = 1 if nz < 1 else nz
            rec = (rec[1], rec[1]+self.nx*self.ny*self.nz*self.nt)
            self.vars[var] = recods(rec, nz)

class Grds:
    """ 读取grd数据 """
    def __init__(self, ctlFileName, fileName):
        self.fileName = fileName
        self._ctl = ControlFile(ctlFileName)
        self.nx = self._ctl.nx
        self.ny = self._ctl.ny
        self.nz = self._ctl.nz
        self.nt = self._ctl.nt
        self.vars = self._ctl.vars

    def read(self, name):
        ''' 读取变量，包含所有维度 '''
        name = name.upper()
        if not name in self.vars:
            if name == "XLONG":
                name = [i for i in self.vars if "LON" in i][0]
            elif name == "XLAT":
                name = [i for i in self.vars if "LAT" in i][0]
            else:
                print(name, "is a wrong name")
                exit()
        beg, end = self.vars[name].rec
        data = np.memmap(self.fileName, dtype="<f", mode="r", offset=beg*4, shape=end-beg)
        # ">f" : > 大小端问题; fortran direct and unformated 输出与C输出一样；
        print("data shape:", data.shape, type(data))
        return self.reshape_mathod(data)

    # 对一维的数据进行reshape，不同的数据需要自定义不同的reshape方式
    def reshape_mathod(self, data):
        y_line = []
        now_index = 0
        for y in range(32):
            d_line = []
            for d in range(180):
                lon_line = []
                for lon in range(54):
                    lat_line = []
                    for lat in range(87):
                        lat_line.append(data[now_index])
                        now_index += 1
                    lon_line.append(lat_line)
                d_line.append(lon_line)
            y_line.append(d_line)
        data = np.array(y_line)
        print("#"*20, "data shape {}".format(data.shape))
        return data

    def read_origin(self, name):
        ''' 读取变量，包含所有维度 '''
        name = name.upper()
        if not name in self.vars:
            if name == "XLONG":
                name = [i for i in self.vars if "LON" in i][0]
            elif name == "XLAT":
                name = [i for i in self.vars if "LAT" in i][0]
            else:
                print(name, "is a wrong name")
                exit()
        beg, end = self.vars[name].rec
        data = np.memmap(self.fileName, dtype="<f", mode="r", offset=beg*4, shape=end-beg)
        # ">f" : > 大小端问题; fortran direct and unformated 输出与C输出一样；
        print("data shape:", data.shape, type(data))
        return np.array(data)

    # def plot(self, name: str, levle: int =0):
    def plot(self, name, level=0, lat=None, lon=None, values=None):
        ''' 快速查看某些变量 '''
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        data = self.read(name)[level, :, :]
        if lat:
            plt.contourf(lon, lat, data, levels=range(0, 30, 3))
        else:
            if not values:
                plt.contourf(data)
            else:
                plt.contourf(data, levels=values)
        plt.grid()
        plt.colorbar()
        plt.savefig(name+".png")

    # def plot(self, name: str, levle: int =0):
    def plot_single_frame(self, frame, name="", level=0, lat=None, lon=None, values=None):
        ''' 快速查看某些变量 '''
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # data = self.read(name)[level, :, :]
        data = frame
        if lat:
            plt.contourf(lon, lat, data, levels=range(0, 30, 3))
        else:
            if not values:
                plt.contourf(data)
            else:
                plt.contourf(data, levels=values)
        plt.grid()
        plt.colorbar()
        plt.savefig(name + ".png")

def read_bin_to_numpy():
    f = Grds(ens_mean_file, fileName)
    data = f.read("tmax")
    return data

def unit_test():
    # read_from_ctl(fileName)
    a = Grds(ens_mean_file, fileName)
    a_out = a.read("tmax")
    print(a_out.shape, type(a_out))
    single_fram = a_out[:,:,0,0]
    print(single_fram.shape)
    # a.plot_single_frame(single_fram, "single frame show")
    print(single_fram.shape)
    store_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"
    f_store = h5py.File(store_file_path, 'w')
    f_store["bin_label"] = a_out
    f_store.close()

def main():
    a = Grds(ens_mean_file, fileName)
    a_out = a.read("tmax")
    print(a_out.shape, type(a_out))
    single_fram = a_out[:, :, 0, 0]
    print(single_fram.shape)
    # np.savetxt('single_frame.csv', single_fram)
    # a.plot_single_frame(single_fram, "single frame show")
    print(single_fram.shape)

if __name__ == '__main__':
    # main()
    unit_test()