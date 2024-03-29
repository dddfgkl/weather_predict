import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from construct_dataset import read_h5
from read_nc import desc_single_ncFile
from matplotlib.backends.backend_pdf import PdfPages

# plt.switch_backend('Agg')

def desc_h5_file():
    raw_cpc_file_path = "/home/datanfs/macong_data/32year_180day_cpc_data_filled.h5"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"

    train_path_macong = "/home/datanfs/macong_data/train_daqisuo.h5"
    val_path_macong = "/home/datanfs/macong_data/valid_daqisuo.h5"

    train_path_zhulifa = "/home/zhulifa/python-dev/macong_data/train_daqisuo.h5"
    val_path_zhulifa = "/home/zhulifa/python-dev/macong_data/valid_daqisuo.h5"

    raw_cpc = read_h5(train_path_macong, "data")
    raw_data = read_h5(train_path_macong, "label")

    print("#" * 10, "basic info")
    # transpose the data to the shape you want
    # present shape is 87x54x180x32
    # raw_cpc = raw_cpc.transpose(3, 2, 1, 0)
    # raw_data = raw_data.transpose(3, 2, 0, 1)
    print(raw_cpc.shape)
    print(raw_data.shape)

    for x in range(4800):
        for y in range(6):
            for lat in range(54):
                for lot in range(87):
                    if raw_cpc[x][y][lat][lot][0] < -100 or raw_cpc[x][y][lat][lot][0] > 100:
                        print("too low or too high")

    return raw_cpc, raw_data

def plot_origin_data_test():
    cpc_1981_path = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1981.nc"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"

    cpc_data = desc_single_ncFile(cpc_1981_path, 'tmax')
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print(cpc_data.shape)
    print(raw_data.shape)

    raw_data = raw_data[0]

    for d in range(180):
        for lat in range(54):
            for lon in range(87):
                if cpc_data[d][lat][lon] < -100 or np.isnan(cpc_data[d][lat][lon]):
                    cpc_data[d][lat][lon] = raw_data[d][lat][lon]
    mse = []
    x = [i for i in range(180)]
    for d in range(180):
        mse.append(sklearn_MSE(raw_data[d], cpc_data[d]))
    print(mse)
    print(x)

    plot_graph(x, mse, 'bin_inter_day.jpg')

    print("plot test over")


def plot_center():
    x = [i for i in range(180)]
    cpc, bin = desc_h5_file()
    year_mse = []
    # mse = []
    for y in range(32):
        mse = []
        for d in range(180):
            mse.append(sklearn_MSE(cpc[y][d], bin[y][d]))
            # print(f"year {y} , day {d}, shape {cpc[y][d].shape}")
            # plot_image(cpc[y][d])
            # plot_image(bin[y][d])
        # year_mse.append(sum(mse)/len(mse)
        plot_graph(x, mse)
        # break
    # mse = []

def plot_image_test():
    cpc_1981_path = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1981.nc"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_label_data.h5"

    cpc_data = desc_single_ncFile(cpc_1981_path, 'tmax')
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print(cpc_data.shape)
    print(raw_data.shape)

    raw_data = raw_data.transpose(3, 2, 1, 0)[0]

    for d in range(180):
        for lat in range(54):
            for lon in range(87):
                if cpc_data[d][lat][lon] < -100 or np.isnan(cpc_data[d][lat][lon]):
                    cpc_data[d][lat][lon] = raw_data[d][lat][lon]
    mse = []
    x = [i for i in range(180)]
    select_day = [0,1,2,3,4,5,6,7,8,9,10,175,176,177,178,179]
    for d in range(180):
        if d not in select_day:
            continue
        mse.append(sklearn_MSE(cpc_data[d], raw_data[d]))
        plot_image(cpc_data[d], f"day{d}_cpc_image.jpg")
        plot_image(raw_data[d], f"day{d}_bin_image.jpg")

    print(mse)
    print(x)

    # plot_graph(x, mse, './')

    print("plot test over")

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
def plot_graph(x, y, file_name=None):
    plt.figure()
    # x = [1,2,3,4,5]
    # y = [2,3,1,5,7]

    # implement your awesome plot
    plt.xlim()
    plt.ylim()

    plt.xlabel("x(year)")
    plt.ylabel("y(mse)")

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title("year-mse plot single year")
    plt.show()

    if file_name != None:
        plt.savefig(file_name)
    else:
        # plt.savefig('./test1.jpg')p
        pass

def plot_image(matrix, file_name=None):
    # plt.xlim()
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title("day matrix")

    plt.matshow(matrix)
    if file_name == None:
        # plt.savefig('test.jpg')
        pass
    else:
        plt.savefig(file_name)


# 以图像方式画图,测试在发开机上不能正常显示
def plot_image_one(image, fileName=None):
    im = Image.fromarray(image)
    im.show()
    # plt.show()
    # plt.close()

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
    print("test!git")
    # plot_graph('./')
    # unit_test()
    # plot_center()
    desc_h5_file()
    # plot_image_test()