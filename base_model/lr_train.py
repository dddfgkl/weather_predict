import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import h5py

def lr_train_test(X_train,y_train,X_test,y_test):
    lr = LinearRegression().fit(X_train, y_train)
    pred=lr.predict(X_test)
    pred=np.where(pred>0,pred,0)
    df_jieguo = pd.DataFrame(y_test,columns=["pm25"])
    df_jieguo = df_jieguo.reset_index()
    df_jieguo['pred'] = pred
    return pred


if  __name__=='__main__':
    data_path="./LRmodel"
    ftrain="train_daqisuo_lr.h5"
    fvaild="valid_daqisuo_lr.h5"
    ftest="test_daqisuo_lr.h5"
    grid_pred=[]
    
    read_train = h5py.File(ftrain,'r')
    #read_valid = h5py.File(fvalid,'r')
    read_test = h5py.File(ftest,'r')
    for r in range(339):
        for c in range(432):
            X_train = read_train['data'][:,:,r,c]
            y_train = read_train['label'][:,:,r,c]
            X_test = read_test['data'][:,:,r,c]
            y_test = read_test['label'][:,:,r,c]
            pred=lr_train_test(X_train,y_train,X_test,y_test)
            grid_pred.append(pred)
    read_train.close()
    read_test.close()
    for i in range(1600):# test sample numbers
        grid=[]
        for j in range(len(grid_pred)):
            grid.append(grid_pred[j][i])
        grid=np.asarray(grid,dtype=np.float32)      
        grid.reshape(339,432)
        np.save("./output/LRPred/lr_"+str(i+1),grid)
