''' grd文件接口 '''

from collections import namedtuple

import numpy as np

class ControlFile:
    '''
    解析grd的控制文件
    '''
    def __init__(self, ctlFileName):
        self.nx = 0
        self.ny = 0
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
            if line[:1].isalpha() and len(cols) > 1:
                keyName = cols[0].upper()
                varsLst.append(keyName)
                infoDic[keyName] = cols[1:]

        self.get_mete(infoDic, varsLst)

    def get_mete(self, infoDic, varsLst):
        ''' 解析配置文件的信息 '''
        if  "PDEF" in infoDic.keys():
            self.nx = int(infoDic["PDEF"][0])
            self.ny = int(infoDic["PDEF"][1])
        else:
            self.nx = int(infoDic["XDEF"][0])
            self.ny = int(infoDic["YDEF"][0])

        recods = namedtuple("recods", "rec  nz")
        rec = (0, 0) # conventionally the end of rec is not used
        for var in varsLst[varsLst.index("VARS")+1:]:
            nz = int(infoDic[var][0])
            nz = 1 if nz < 1 else nz
            rec = (rec[1], rec[1]+self.nx*self.ny*nz)
            self.vars[var] = recods(rec, nz)

class Grds:
    """ 读取grd数据 """
    def __init__(self, ctlFileName, fileName):
        self.fileName = fileName
        self._ctl = ControlFile(ctlFileName)
        self.nx = self._ctl.nx
        self.ny = self._ctl.ny
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
        data = np.memmap(self.fileName, dtype=">f", mode="r", offset=beg*4, shape=end-beg)
        # ">f" : > 大小端问题; fortran direct and unformated 输出与C输出一样；
        return np.array(data).reshape(self.vars[name].nz, self.ny, self.nx,)

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
