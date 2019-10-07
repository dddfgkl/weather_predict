import torch
import h5py
import os
import numpy as np
from netCDF4 import Dataset
# from model.readNctoh5 import readNC


fileName1 = "../data/CN-Reanalysis2017101907.nc"
fileName2 = "../data/tmax.1981.nc"
bin_file = ""
dataPath = ""
dir_path = "/home/machong/PM25-work/CPC_global/temp"

store_path = "/home/machong/PM25-work/Hind3_daily/180day_data.h5"
store_path2 = "/home/datanfs/macong_data/100day_data.h5"
# 闰年
month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30 , 31, 30, 31]

# desc a single nc file
def desc_single_ncFile(fileName, wantedKeys):
    nc_obj = Dataset(fileName)
    print("nc variable keys ", nc_obj.variables.keys())
    keys = nc_obj.variables.keys()
    for key in keys:
        print(key, nc_obj.variables[key][:].shape)
        if len(nc_obj.variables[key][:].shape) > 1:
            print("length is bigger than one, ", len(nc_obj.variables[key][:].shape))
            continue
        print(nc_obj.variables[key][:10].data)
    data = []
    if wantedKeys != None:
        for key in wantedKeys:
            if key in nc_obj.variables.keys():
                data.append(nc_obj.variables[key][:].data)
    nc_obj.close()
    return data


def desc_single_nc_detail(fileName):
    nc_obj = Dataset(fileName)
    longitude = nc_obj.variables["lon"][:]
    latitude = nc_obj.variables["lat"][:]

    # basic variable
    print("latitude:",latitude.shape)
    print("lonhitude:", longitude.shape)

    print("latitude degree", latitude[:10])
    print("longitude degree", longitude[:10])


def readNc2h5(savepth, filepth, f):
    # data = readNC(path)
    h5file = h5py.File(os.path.join(savepth, f[:-3] + '.h5'), 'w')
    res = readNC(os.path.join(filepth, f))
    print(res.shape)
    h5file.create_dataset(f[:-3], data=res)
    h5file.close()

def readH5(h5path, f):
    readFile = h5py.File(os.path.join(h5path, f), 'r')
    print(readFile, type(readFile))
    testSet = readFile[f[:-3]]
    print(testSet, type(testSet))
    dataset = readFile[f[:-3]][:]
    print(dataset.shape, type(dataset))

def desc_all_ncFile(dir_path):
    files = os.listdir(dir_path)
    all_available_files = []
    for file in files:
        if file == "README":
            continue
        all_available_files.append(file)
        print(file, type(file))
    sorted(all_available_files)
    all_available_files = sorted(all_available_files)
    for i, file in enumerate(all_available_files):
        print("-------file Name--------", file)
        datas = desc_single_ncFile(os.path.join(dir_path, file), ("tmax", "lat", "lon", "time"))
        print("\n\n")
    print("sum of the nc file ", len(all_available_files))
    print("over ")

# 判断是否是闰年
def is_leap_year(year):
    if year % 100 == 0:
        if year % 400 == 0:
            return True
        else:
            return False
    if year % 4 == 0:
        return True
    return False

# 从原始CPC的nc数据中，抽取对应天数，拼接各个年份数据组织成(经度X纬度X天数X年份)的h5数据存储
def extract_year_from_nc_to_h5(dir_path, store_path):
    if dir_path == None or store_path == None:
        raise FileExistsError
    f = h5py.File(store_path, 'w')
    files = os.listdir(dir_path)
    all_available_files = []
    for file in files:
        if file == "README":
            continue
        all_available_files.append(file)
        print(file, type(file))
    sorted(all_available_files)
    all_available_files = sorted(all_available_files)
    h5_data = []
    for i, file in enumerate(all_available_files):
        print("-------file Name--------", file)
        data, lat, lon, time = desc_single_ncFile(os.path.join(dir_path, file), ("tmax", "lat", "lon", "time"))
        if i == 0:
            f["lat"] =lat
            f["lon"] = lon

        # 3月5日到9月1日，从后向前避免处理闰年，纬度转换成经度X纬度X天数
        data = data[-301:-121, :, :].transpose(2, 1, 0)
        h5_data.append(data)
    h5_data = np.array(h5_data)
    f["tmax"] = h5_data
    print("h5_data shape is : ", h5_data.shape)
    f.close()

# 测试h5文件写入结果
def desc_single_h5_file(h5_file_path):
    if os.path.exists(h5_file_path) == False:
        raise FileExistsError
    f = h5py.File(h5_file_path, 'r')
    print(f.keys())
    for key in f.keys():
        print(key, f[key].shape, type(f[key]), type(f[key][:]), f[key][:].shape)
    print("desc over all keys")

def unit_test():
    # extract_year_from_nc_to_h5(dir_path, store_path2)'
    desc_single_h5_file(store_path2)

def main():
    unit_test()
    # data = desc_single_ncFile(fileName1, "pm25")
    # print(data)
    # desc_single_ncFile(fileName2)
    # desc_single_nc_detail(fileName2)
    '''
    savepth = dataPath
    filepth = dataPath
    f = "CN-Reanalysis2017101907.nc"
    readNc2h5(savepth, filepth, f)
    '''
    # readH5(dataPath, f="CN-Reanalysis2017101907.h5")
    # desc_all_ncFile(dir_path)
    print("main thread over")

if __name__ == '__main__':
    main()