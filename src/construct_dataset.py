import h5py
import os
import numpy as np

def read_h5(file_path, key):
    if os.path.exists(file_path) == False:
        print("file not exist")
        raise FileExistsError
    f = h5py.File(file_path, 'r')
    return f[key][:]

def desc_h5():
    train_path = '/home/datanfs/macong_data/train_daqisuo.h5'
    valid_path = '/home/datanfs/macong_data/valid_daqisuo.h5'

    train_path_52 = "/home/zhulifa/PM25-work/daqisuo_data/train_daqisuo.h5"
    valid_path_52 = "/home/zhulifa/PM25-work/daqisuo_data/valid_daqisuo.h5"

    #train_path = "/home/datanfs/anhui/PM25Pred/train_daqisuo.h5"
    #valid_path = "/home/datanfs/anhui/PM25Pred/valid_daqisuo.h5"

    train_data = read_h5(train_path_52, 'data')
    valid_data = read_h5(valid_path_52, 'data')
    for i in range(4900):
        for d in range(6):
            if np.any(np.isnan(train_data[i][d])):
                print("##### have nan number #####")
            else:
                print("##### have not nan #####")

    print(train_data.shape)
    print(valid_data.shape)

def fill_cpc_data():
    raw_cpc_file_path = "/home/datanfs/macong_data/32year_180day_cpc_data_not_filled.h5"
    raw_data_file_path = "/home/datanfs/macong_data/180day_bin2h5_predict_data.h5"

    filled_cpc_file_path = "/home/datanfs/macong_data/32year_180day_cpc_data_filled.h5"

    raw_cpc = read_h5(raw_cpc_file_path, "cpc")
    raw_data = read_h5(raw_data_file_path, "bin_label")

    filled_store = h5py.File(filled_cpc_file_path, 'w')
    raw_cpc = raw_cpc.transpose(3,2,1,0)
    print("cpc data shape, ", raw_cpc.shape)
    print("bin data label shape, ", raw_data.shape)
    filled_data = np.zeros(raw_cpc.shape)
    print(filled_data.shape)

    nan_other = -100

    count_cpc = 0
    how_many_nan = 0
    for y in range(32):
        print(f"now is process {y} year")
        for d in range(180):
            for lat in range(54):
                for lon in range(87):
                    # print(raw_data[lon][lat][d][y], type(raw_data[lon][lat][d][y]), raw_cpc[lon][lat][d][y], type(raw_cpc[lon][lat][d][y]))
                    if np.isnan(raw_cpc[lon][lat][d][y]) or nan_other > raw_cpc[lon][lat][d][y]:
                        # ("##### raw cpc {} is nan #####".format(raw_cpc[lon][lat][d][y]))
                        how_many_nan += 1
                        filled_data[lon][lat][d][y] = raw_data[lon][lat][d][y]
                    else:
                        filled_data[lon][lat][d][y] = raw_cpc[lon][lat][d][y]
                        count_cpc += 1
                        # print("##### filled with cpc #####")
    print("{} nan number, {} cpc raw number, {} total number".format(how_many_nan, count_cpc, 32*180*54*87))
    filled_store['cpc'] = filled_data
    print("filled data desc, ", filled_data.shape)
    print("fill data complected")

def construct_data(window = 6):
    # read origin data file
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
    raw_cpc = raw_cpc.transpose(3,2,0,1)
    raw_data = raw_data.transpose(3,2,0,1)
    print(raw_cpc.shape)
    print(raw_data.shape)

    # state the path to store data
    h5train = h5py.File(r'/home/datanfs/macong_data/train_daqisuo_v2.h5', 'w')
    h5val = h5py.File(r'/home/datanfs/macong_data/valid_daqisuo_v2.h5', 'w')
    # h5test = h5py.File(r'./test_daqisuo.h5', 'w')

    totol_label_nums = 32 * (180 - window + 1)
    train_sample_nums = 28 * (180 - window + 1)
    val_sample_nums = 4 * (180 - window + 1)
    # test_sample_nums = 0


    train_input_shape = (train_sample_nums, window, 87, 54, 1)
    train_target_shape = (train_sample_nums, 1, 87, 54)
    val_input_shape = (val_sample_nums, window, 87, 54, 1)
    val_target_shape = (val_sample_nums, 1, 87, 54)
    # test_input_shape = (1600, window * 2 + 1, 87, 54, 1)
    # test_target_shape = (1600, 1, 87, 54)

    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    # h5test.create_dataset("data", test_input_shape, np.float32)
    # h5test.create_dataset("label", test_target_shape, np.float32)

    # total_label_nums = 32 * (180 - window * 2 + 1)
    train_cn = 0
    val_cn = 0
    test_cn = 0
    if window % 2 != 0:
        window = window // 2
        for y in range(32):
            print(f"{y}, total 32")
            for d in range(window,180-window):
                if train_cn < train_sample_nums:
                    h5train["data"][train_cn,...] = raw_data[:][:][d-window:d+window+1][y].transpose(2, 0, 1).reshape(2*window+1, 87,54,2*window+1,1)
                    h5train["label"][train_cn, ...] = raw_cpc[:][:][d][y].transpose(1,2,0).reshape(1,87,54)
                    train_cn += 1
                    continue
                if val_cn < val_sample_nums:
                    h5val["data"][val_cn, ...] = raw_data[:][:][d - window:d + window + 1][y].transpose(2, 0, 1).reshape(2*window+1, 87,54,2*window+1,1)
                    h5val["label"][val_cn, ...] = raw_cpc[:][:][d][y].transpose(1, 2, 0).reshape(1,87,54)
                    val_cn += 1
                    continue
                """
                if test_cn < test_sample_nums:
                    h5test["data"][test_cn, ...] = raw_data[:][:][d - window:d + window + 1][y].transpose(2, 0, 1).reshape(12*window+1, 87,54,2*window+1,1)
                    h5test["label"][test_cn, ...] = raw_cpc[:][:][d][y].transpose(1, 2, 0).reshape(1,87,54)
                    test_cn += 1
                    continue
                """
    else:
        left_window = window // 2
        right_window = window // 2 - 1
        for y in range(32):
            print(f"{y}, total 32")
            for d in range(left_window,180-right_window):
                if train_cn < train_sample_nums:
                    h5train["data"][train_cn,...] = raw_data[y][d-left_window:d+right_window+1].reshape(window, 87,54,1)
                    h5train["label"][train_cn, ...] = raw_cpc[y][d].reshape(1,87,54)
                    train_cn += 1
                    continue
                if val_cn < val_sample_nums:
                    h5val["data"][val_cn, ...] = raw_data[y][d-left_window:d+right_window+1].reshape(window, 87,54,1)
                    h5val["label"][val_cn, ...] = raw_cpc[y][d].reshape(1,87,54)
                    val_cn += 1
                    continue
                """
                if test_cn < test_sample_nums:
                    h5test["data"][test_cn, ...] = raw_data[:][:][d - window:d + window + 1][y].transpose(2, 0, 1).reshape(12*window+1, 87,54,2*window+1,1)
                    h5test["label"][test_cn, ...] = raw_cpc[:][:][d][y].transpose(1, 2, 0).reshape(1,87,54)
                    test_cn += 1
                    continue
                """

    h5train.close()
    h5val.close()
    # h5test.close()

if __name__ == '__main__':
    # construct_data()
    fill_cpc_data()
    # desc_h5()

