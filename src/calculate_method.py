import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from construct_dataset import read_h5
from read_nc import desc_single_ncFile

def desc_h5_file():
    raw_cpc_file_path = "/home/datanfs/macong_data/180day_everyday_label_data_filled_v2.h5"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_label_data.h5"
    raw_cpc = read_h5(raw_cpc_file_path, "cpc")
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print("basic info")
    print("cpc data shape, ", raw_cpc.shape)
    print(raw_cpc[0][0][:][:])
    print("bin data label shape, ", raw_data.shape)
    print(raw_data[:][:][0][0])
    # transpose the data to the shape you want
    # present shape is 87x54x180x32
    raw_cpc = raw_cpc.transpose(3, 2, 0, 1)
    raw_data = raw_data.transpose(3, 2, 0, 1)
    print(raw_cpc.shape)
    print(raw_data.shape)
    return raw_cpc, raw_data

def plot_origin_data_test():
    cpc_1981_path = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1981.nc"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_label_data.h5"

    cpc_data = desc_single_ncFile(cpc_1981_path, 'tmax')
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print(cpc_data.shape)
    print(raw_data.shape)

    print("plot test over")


def plot_center():
    x = [i for i in range(180)]
    cpc, bin = desc_h5_file()
    mse = []
    for y in range(32):
        for d in range(180):
            mse.append(sklearn_MSE(cpc[y][d], bin[y][d]))
        plot_graph(x, mse, './')
        mse = []

# self define mse
def MSE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    dot_count = 1
    for x in y_true.shape:
        dot_count *= x
    print(f"dot cont {dot_count}, shape is {y_pred.shape}")
    mse = np.sum((y_true - y_pred) ** 2) / dot_count
    return mse

# sklearn mse
def sklearn_MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# self define your plot data
def plot_graph(x, y, dir_path, file_name=None):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)
    plt.figure()
    # x = [1,2,3,4,5]
    # y = [2,3,1,5,7]

    # implement your awesome plot
    plt.xlim()
    plt.ylim()

    plt.xlabel("x(day)")
    plt.ylabel("y(mse)")

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title("day-mse plot single year")
    plt.show()

def test_mse():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    sk = mean_squared_error(y_true, y_pred)
    mse = MSE(y_true, y_pred)
    print(f"{sk}  {mse}")

def unit_test():
    # test_mse()
    # desc_h5_file()
    plot_origin_data_test()

if __name__ == '__main__':
    # plot_graph('./')
    unit_test()
    # plot_center()