import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from construct_dataset import read_h5
from read_nc import desc_single_ncFile
from matplotlib.backends.backend_pdf import PdfPages
import calculate_method
import h5py

def plot_image_from_raw_data():
    cpc_1981_path = "/home/datanfs/liutao_backup1/Hind3_label/Tmax/tmax_lbl.1983.nc"
    # raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_label_data.h5"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"


    cpc_data = desc_single_ncFile(cpc_1981_path, 'tmax')
    raw_data = read_h5(raw_data_file_path, "bin_label")
    print(cpc_data.shape)
    print(raw_data.shape)

    raw_data = raw_data[2]

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
    select_day = []
    for d in range(180):
        # if d not in select_day:
        #    continue
        mse.append(mean_squared_error(cpc_data[d], raw_data[d]))
        # plot_single_image(cpc_data[d], "1981_cpc_{}_day.jpg".format(d))
        # plot_single_image(raw_data[d], "1981_predict_{}_day.jpg".format(d))

    print(mse)
    print(x)

    calculate_method.plot_graph(x, mse)

    print("plot test over")

def plot_loss_rate(x, y):
    plt.figure()
    # x = [1,2,3,4,5]
    # y = [2,3,1,5,7]

    # implement your awesome plot
    plt.xlim()
    plt.ylim()

    plt.xlabel("epoch")
    plt.ylabel("y")

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title("epoch loss")
    plt.show()


def plot_single_image(matrix, file_name=None):
    plt.switch_backend('agg')
    # plt.xlim()
    # plt.xlabel("lon")
    # plt.ylabel("lat")
    # plt.title("day matrix")
    # plt.title(file_name)

    plt.matshow(np.array(matrix))
    plt.title(file_name, y=-0.1)
    # plt.show()

    # ssh链接跳板机链接开发机的时候不能显示，所以需要存储一下，
    if file_name == None:
        # plt.savefig('test.jpg')
        pass
    else:
        plt.savefig(file_name)

learning_curve_path = '../nn_model/record/train_record.h5'

def read_learning_record():
    x = read_h5(learning_curve_path, 'epoch_h5')
    y = read_h5(learning_curve_path, 'train_loss_h5')
    plot_learning_curve(x, y)
    print("plot over!")


def plot_learning_curve(x, y):
    plt.figure()
    # plt.title("learning")
    plt.scatter(x, y)
    plt.title()
    plt.show()

    pass

def read_origin_single_frame():
    import read_ctl
    a = read_ctl.Grds(read_ctl.ens_mean_file_windows, read_ctl.fileName_windows)
    a_out = a.read_origin("tmax")
    print(a_out.shape)
    first_frame = a_out[:87*54]
    first_frame = first_frame.reshape(54, 87)
    print(first_frame.shape)
    plot_single_image(first_frame)


def read_after_process_data():
    train_path_macong = "/home/datanfs/macong_data/train_daqisuo.h5"
    val_path_macong = "/home/datanfs/macong_data/valid_daqisuo.h5"
    a = read_h5(train_path_macong, 'data')
    b = read_h5(train_path_macong, 'label')
    print("train data shape", a.shape)
    print("train label shape", b.shape)
    c = a[0][0][:][:].transpose(2,0,1)[0]
    print(a[0].shape)
    print(a[0][0].shape)
    print(a[0][0][:][:].shape)
    print(c.shape)
    plot_single_image(c, 'test1.jpg')

def plot_origin_data():
    pass

def plot_center():
    record_epoch = read_h5("/home/zhulifa/python-dev/weather_predict/nn_model/record/train_record.h5", "epoch_h5")
    record_train_loss_h5 = read_h5("/home/zhulifa/python-dev/weather_predict/nn_model/record/train_record.h5", "train_loss_h5")
    # record_train_loss_h5 = read_h5()
    plot_loss_rate(record_epoch, record_train_loss_h5)
    print("hello world")

def plot_processed_data():
    train_path_macong = "/home/datanfs/macong_data/train_daqisuo.h5"
    if os.path.exists(train_path_macong) == False:
        print("file not exist")
        raise FileExistsError
    f = h5py.File(train_path_macong, 'r')
    data = f['data'][:]
    label = f['label'][:]
    print(data.shape)
    print(label.shape)

    # plot method
    cnt = 0
    for x in range(4900):
        print(f"now is process {cnt}")
        plt.figure()
        plt.suptitle('Multi_Image')
        for y in range(6):
            plt.subplot(1, 7, i+1), plt.title('Observe {}'.format(i+1))
            plt.imshow(data[i][0], cmap=plt.cm.gray), plt.axis('off')
        plt.subplot(1, 7, 7), plt.title('Label {}'.format(i + 1))
        plt.imshow(label[i], cmap=plt.cm.gray), plt.axis('off')
        plt.savefig("./outPic/output_" + str(cnt) + ".png")
        cnt += 1
        if cnt > 2:
            break
    print('plot over')

def unit_test():
    # plot_image_from_raw_data()
    # read_origin_single_frame()
    # read_learning_record()
    # read_after_process_data()
    plot_processed_data()

if __name__ == '__main__':
    # plot_center()
    unit_test()