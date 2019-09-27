import numpy as np
import h5py
import os

def splitH5(h5path, h5filename,tPre=6,tNext=1,step=1):
    readFile=h5py.File(os.path.join(h5path,h5filename),'r')
    dataset = readFile[h5filename[:-3]][:]
    kind = dataset.shape[3]  #26
    ms = np.zeros((kind, 2))
    #minmax = np.zeros((kind,2))
    for i in range(kind):
        all_data = dataset[:,:,:,i]
        average = np.average(all_data)
        std = np.std(all_data)
        ms[i,:] = (average, std)
        #minval = np.min(all_data)
        #maxval = np.max(all_data)
        #minmax[i,:] = (minval,maxval)
    tmax=dataset.shape[0]-tPre-tNext-step
    inp,oup=[],[]
    for t in range(tmax):
        # weather, morphology history data, one step ahead
        weather = dataset[t+step: t+tPre+step, :, :, 15:]
        # aqi history data, one step behind
        aqi = dataset[t:t+tPre, :, :, :15]
        input_data = np.concatenate((aqi, weather), -1)
        for i in range(input_data.shape[3]):
            all_data = input_data[:,:,:,i]
            all_data = (all_data - ms[i, 0]) / ms[i, 1]
            #all_data = (all_data - minmax[i, 0]) / (minmax[i, 1] - minmax[i,0])
            input_data[:,:,:,i] = all_data
        # groud truth output data
        output_data = dataset[t+tPre:t+tPre+tNext, :, :, 0]
        inp.append(input_data)
        oup.append(output_data)
    return inp,oup

if __name__=='__main__':
    file_path="./pre_data"
    h5files=sorted(os.listdir(file_path))

    h5train=h5py.File(r'/home/th/anhui/PM25Pred/train.h5', 'w')
    h5val=h5py.File(r'/home/th/anhui/PM25Pred/valid.h5', 'w')
    h5test=h5py.File(r'/home/th/anhui/PM25Pred/test.h5', 'w')
    train_input_shape=(6440,6,269,239,26) #161*59=9499
    train_target_shape=(6440,1,269,239)
    val_input_shape=(1449,6,269,239,26) 
    val_target_shape=(1449,1,269,239)
    test_input_shape=(1610,6,269,239,26) 
    test_target_shape=(1610,1,269,239)
    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    h5test.create_dataset("data", test_input_shape, np.float32)
    h5test.create_dataset("label", test_target_shape, np.float32)
    cnt=0
    for f in h5files[:40]: 
        inp,oup=splitH5(file_path,f) 
        for i,o in zip(inp,oup):
            h5train["data"][cnt, ...] = i
            h5train["label"][cnt, ...] = o
            cnt+=1
    cnt=0
    for f in h5files[40:49]: 
        inp,oup=splitH5(file_path,f) 
        for i,o in zip(inp,oup):
            h5val["data"][cnt, ...] = i
            h5val["label"][cnt, ...] = o
            cnt+=1
    cnt=0
    for f in h5files[49:]: 
        inp,oup=splitH5(file_path,f) 
        for i,o in zip(inp,oup):
            h5test["data"][cnt, ...] = i
            h5test["label"][cnt, ...] = o
            cnt+=1
    h5train.close()
    h5val.close()
    h5test.close()
