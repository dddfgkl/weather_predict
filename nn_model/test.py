import h5py
import torch
import numbers
import os,sys
import random
import time
import numpy as np
import torch.autograd as autograd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import tensorwatch as tw
import torchvision
from torchvision import transforms
from dataset import AirDataset,H5Dataset
from torchsummary import summary
#from tensorboardX import SummaryWriter
from conf import settings
from utils import *
import WarmUpLR
from model_complexity import compute_model_complexity
from modelsize_estimate import compute_modelsize
from gpu_mem_track import  MemTracker

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)

from airConvlstm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



def test(sample_batched, model,criterion,cnt=None):
    input_data = Variable(sample_batched['input'].permute(0,1,4,2,3).float().to(device))
    label = Variable(sample_batched['output'].squeeze().float().to(device))
    model.to(device)
    model.eval()
    out = model(input_data).squeeze() #(batch,339,432)
    if cnt:
        plt.figure(figsize=(10,10))
        plt.suptitle('Multi_Image')
        plt.subplot(1,2,1), plt.title('Label')
        plt.imshow(label.cpu().numpy(),cmap=plt.cm.gray), plt.axis('off')
        plt.subplot(1,2,2), plt.title('Pred')
        plt.imshow(out.cpu().numpy(),cmap=plt.cm.gray), plt.axis('off')
        plt.savefig("./outPic/output_"+str(cnt)+".png")
    loss = criterion(out, label)
    error = float(loss.item())
    return error

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


#headers = ['TPM25', 'SO2', 'NO2', 'CO', 'O3','ASO4', 'ANO3', 'ANH4', 'BC', 'OC','PPMF','PPMC','SOA','TPM10','O3_8H','U','V','T','P','HGT','RAIN','PBL','RH','VISIB','AOD','EXT']
headers=["pm25", "pm10", "so2", "no2", "co", "psfc", "u", "v", "temp", "rh"]
batch_size = 1

#readFile = h5py.File('./pre_data/2018010116.h5','r')
#dataset = readFile['2018010116'][:] #shape is (169,269,239,26)

#trainingset = AirDataset(dataset[:120]) #8281
#validationset = AirDataset(dataset[120:])
#loader_train = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
#loader_valid = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
#loader_test = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)



train_path = "./train_daqisuo.h5"
val_path = "./valid_daqisuo.h5"
test_path = "./test_daqisuo.h5"

h5train = H5Dataset(train_path)
h5val = H5Dataset(val_path)
h5test = H5Dataset(test_path)

loader_train =  DataLoader(h5train, batch_size=1,shuffle=False,num_workers=1)
loader_valid =  DataLoader(h5val, batch_size=1,shuffle=False,num_workers=1)
loader_test =  DataLoader(h5test, batch_size=1,shuffle=False,num_workers=1)


# height = 339 #269
# width = 432 #239

height = 54 #269
width = 87 #239
input_dim = 10 #26
n_layer = 2
hidden_size = [64, 128]
output_dim = 1
n_epoch = 1000
learning_rate = 1e-4
weight_decay = 0.9
weight_decay_epoch = 10
direc = './convlstm/model/notebook_daqisuo/'
if not os.path.exists(direc):
    os.makedirs(direc)

air = AirConvLSTM(input_size=(height, width),
                    input_dim=len(headers),
                    hidden_dim=hidden_size,
                    kernel_size=(3, 3),
                    num_layers=n_layer,
                    output_dim=output_dim,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)

if torch.cuda.device_count() > 1:
    air  = nn.DataParallel(air)
air.to(device)

#num_params,flops=compute_model_complexity(air, (2,6,10,339,432), verbose=True) #cpu
#compute_modelsize(air,torch.zeros((2,6,10,339,432)))
#summary(air, (6,10,339,432))
#tw.draw_model(air,[1,6,10,339,432])
#tw.model_states(air,[1,6,10,339,432])


optimizer = optim.Adam(air.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)

resume=True
if resume:
    path=r"convlstm/model/notebook_daqisuo/new_epoch_572.pt"
    checkpoint = torch.load(path)
    air.load_state_dict(checkpoint['model'])
avgError=0
cnt=0
with torch.no_grad():
    for i_batch, sample_batched in enumerate(loader_train,1):
        cnt+=1
        test_error = test(sample_batched, air, nn.MSELoss(),i_batch)
        print('test loss = {}'.format(test_error))
        avgError+=test_error
print("avg loss = ", avgError//cnt)
