import h5py
import os
import numpy as np

def read_h5(file_path, key):
    if os.path.exists(file_path) == False:
        raise FileExistsError
    f = h5py.File(file_path, 'r')
    return f[key][:]

def construct_data(window = 2):
    # read origin data file
    raw_cpc = read_h5("", "cpc")
    raw_data = read_h5("", "tmax")

    # transpose the data to the shape you want
    # present shape is 87x54x180x32
    raw_cpc = raw_cpc.transpose()
    raw_data = raw_data.transpose()

    # state the path to store data
    h5train = h5py.File(r'./train_daqisuo.h5', 'w')
    h5val = h5py.File(r'./valid_daqisuo.h5', 'w')
    # h5test = h5py.File(r'./test_daqisuo.h5', 'w')

    totol_label_nums = 32 * (180 - window * 2 + 1)
    train_sample_nums = 0
    val_sample_nums = 0
    test_sample_nums = 0


    train_input_shape = (3200, window * 2 + 1, 87, 54, 1)
    train_target_shape = (3200, 1, 87, 54)
    val_input_shape = (1024, window * 2 + 1, 87, 54, 1)
    val_target_shape = (1024, 1, 87, 54)
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
    for y in range(32):
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

    h5train.close()
    h5val.close()
    # h5test.close()



