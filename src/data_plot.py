import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from construct_dataset import read_h5
from read_nc import desc_single_ncFile
from matplotlib.backends.backend_pdf import PdfPages

def plot_image_from_raw_data():
    cpc_1981_path = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1981.nc"
    # raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_label_data.h5"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"


    cpc_data = desc_single_ncFile(cpc_1981_path, 'tmax')
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print(cpc_data.shape)
    print(raw_data.shape)

    raw_data = raw_data.transpose()[0]

    # 数据清洗，插值完的cpc数据中依旧有脏数据，脏数据包括nan值和极小值
    for d in range(180):
        for lat in range(54):
            for lon in range(87):
                if cpc_data[d][lat][lon] < -100 or np.isnan(cpc_data[d][lat][lon]):
                    cpc_data[d][lat][lon] = raw_data[d][lat][lon]
    mse = []
    x = [i for i in range(180)]

    # 选择部分天数进行画图
    select_day = [0,1,2,3,4,5,6,7,8,9,10,175,176,177,178,179]
    for d in range(180):
        if d not in select_day:
            continue
        mse.append(mean_squared_error(cpc_data[d], raw_data[d]))
        plot_single_image(cpc_data[d], "1981_cpc_{}_day.jpg".format(d))
        plot_single_image(raw_data[d], "1981_predict_{}_day.jpg".format(d))

    print(mse)
    print(x)

    # plot_graph(x, mse, './')

    print("plot test over")

def plot_single_image(matrix, file_name=None):
    plt.switch_backend('agg')
    # plt.xlim()
    # plt.xlabel("lon")
    # plt.ylabel("lat")
    # plt.title("day matrix")

    plt.matshow(np.array(matrix))
    # plt.show()

    # ssh链接开发机的时候不能显示，所以需要存储一下，
    if file_name == None:
        # plt.savefig('test.jpg')
        pass
    else:
        plt.savefig(file_name)

def read_origin_single_frame():
    import read_ctl
    a = read_ctl.Grds(read_ctl.ens_mean_file_windows, read_ctl.fileName_windows)
    a_out = a.read_origin("tmax")
    print(a_out.shape)
    first_frame = a_out[:87*54]
    first_frame = first_frame.reshape(54, 87)
    print(first_frame.shape)
    plot_single_image(first_frame)

def plot_origin_data():
    pass

def unit_test():
    plot_image_from_raw_data()
    # read_origin_single_frame()

if __name__ == '__main__':
    unit_test()