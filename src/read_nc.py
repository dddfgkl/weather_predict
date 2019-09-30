import torch
import h5py
import os
from netCDF4 import Dataset
from model.readNctoh5 import readNC


fileName1 = "../data/CN-Reanalysis2017101907.nc"
fileName2 = "../data/tmax.1981.nc"
bin_file = ""
dataPath = ""
dir_path = "/home/machong/PM25-work/CPC_global/temp"

# desc a single nc file
def desc_single_ncFile(fileName, wanted_key="tmax"):
    nc_obj = Dataset(fileName)
    print("nc variable keys ", nc_obj.variables.keys())
    keys = nc_obj.variables.keys()
    for key in keys:
        print(key, nc_obj.variables[key][:].shape)
        if len(nc_obj.variables[key][:].shape) > 1:
            print("length is bigger than one, ", len(nc_obj.variables[key][:].shape))
            continue
        print(nc_obj.variables[key][:10])
    data = []
    if wanted_key in nc_obj.variables.keys():
        data = nc_obj.variables[wanted_key][:].data
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
    for file in all_available_files:
        print("-------file Name--------", file)
        data = desc_single_ncFile(os.path.join(dir_path, file))
        if file == "tmax.1981.nc":
            a = data[0][0][0]
            print("xxxxxxx  ", a)
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


def main():
    data = desc_single_ncFile(fileName1, "pm25")
    print(data)
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