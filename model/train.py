import h5py
import torch
import numbers
import os,sys
import random
import time
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
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


torch.manual_seed(2018)
torch.cuda.manual_seed(2018)

from airConvlstm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def train(sample_batched, model, optimizer, criterion):
    #print("sample_batched shape: ",sample_batched['input'].shape, sample_batched['output'].shape)
    input_data = Variable(sample_batched['input'].permute(0,1,4,2,3).float().to(device))
    label = Variable(sample_batched['output'].squeeze().float().to(device))
    model.to(device)
    model.train()
    out = model(input_data).squeeze()
    optimizer.zero_grad()
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    
    error = float(loss.item())
    return error

def valid(sample_batched, model, criterion):
    input_data = Variable(sample_batched['input'].permute(0,1,4,2,3).float().to(device))
    label = Variable(sample_batched['output'].squeeze().float().to(device))
    model.to(device)
    model.eval()
    out = model(input_data).squeeze()
    loss = criterion(out, label)
    error = float(loss.item())
    return error

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


#headers = ['TPM25', 'SO2', 'NO2', 'CO', 'O3','ASO4', 'ANO3', 'ANH4', 'BC', 'OC','PPMF','PPMC','SOA','TPM10','O3_8H','U','V','T','P','HGT','RAIN','PBL','RH','VISIB','AOD','EXT']
headers=["pm25", "pm10", "so2", "no2", "co", "psfc", "u", "v", "temp", "rh"]
batch_size = 16

#readFile = h5py.File('./pre_data/2018010116.h5','r')
#dataset = readFile['2018010116'][:] #shape is (169,269,239,26)

#trainingset = AirDataset(dataset[:120]) #8281
#validationset = AirDataset(dataset[120:])
#loader_train = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
#loader_valid = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
#loader_test = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

train_path_macong = "/home/datanfs/macong_data/train_daqisuo.h5"
val_path_macong = "/home/datanfs/macong_data/valid_daqisuo.h5"

train_path_52 = "/home/zhulifa/PM25-work/daqisuo_data/train_daqisuo.h5"
val_path_52 = "/home/zhulifa/PM25-work/daqisuo_data/valid_daqisuo.h5"

train_path = "/home/datanfs/anhui/PM25Pred/train_daqisuo.h5"
val_path = "/home/datanfs/anhui/PM25Pred/valid_daqisuo.h5"
test_path = "./test_daqisuo.h5"

print("##### start load dataset #####")
h5train = H5Dataset(train_path_macong)
h5val = H5Dataset(val_path_macong)
# h5test = H5Dataset(test_path)

loader_train = DataLoader(h5train, batch_size=batch_size,shuffle=True,num_workers=16,drop_last=True)
loader_valid = DataLoader(h5val, batch_size=batch_size,shuffle=True,num_workers=16,drop_last=True)
# loader_test =  DataLoader(h5test, batch_size=1,shuffle=False,num_workers=16)
print("##### load dataset over #####")


#height = 339
#width = 432

height = 87 #269
width = 54 #239
headers_length = 1
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

record_dir = "./record/"
record_file_name = "train_record.h5"
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
file_record = h5py.File(os.path.join(record_dir, record_file_name), 'w')

air = AirConvLSTM(input_size=(height, width),
                    input_dim=headers_length,
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
start = time.time()

min_loss = np.inf
start_epoch=1
resume=False

epoch_h5 = []
train_loss_h5 = []
train_lr_h5 = []
time_cost_h5 = []
validation_loss_h5 = []

if resume:
    path=r"convlstm/model/notebook/epoch_128.pt"
    checkpoint = torch.load(path)
    air.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch=checkpoint['epoch']+1
print("##### start train, epoch is {} #####".format(n_epoch))
for epoch in range(start_epoch, n_epoch + 1):
    if epoch > weight_decay_epoch:
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_loss = 0
    j = 0
    for i_batch, sample_batched in enumerate(loader_train):
        train_error = train(sample_batched, air, optimizer, nn.MSELoss())
        train_epoch_loss += train_error
    train_loss = train_epoch_loss//len(loader_train)
    end = time.time()
    time_cost = sec_to_hms(int(end-start))
    print('epoch = {}, training loss = {}, lr = {}, time cost = {}, length loader_train = {}'.format(epoch, train_loss, lr, time_cost, len(loader_train)))

    epoch_h5.append(epoch)
    train_loss_h5.append(train_loss)
    train_lr_h5.append(lr)
    time_cost_h5.append(time_cost)
    
    val_epoch_loss = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(loader_valid):
            val_error = valid(sample_batched, air, nn.MSELoss())
            val_epoch_loss += val_error
    val_loss = val_epoch_loss//len(loader_valid)
    print('validation loss = {}'.format(val_loss))
    validation_loss_h5.append(val_loss)
    
    if val_loss < min_loss:
        min_loss = val_loss
        name = direc + 'new_epoch_'+str(epoch)+'.pt'
        print('saving model to {}'.format(name))
        state = { 'model': air.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, name)

file_record["epoch_h5"] = epoch_h5
file_record["train_loss_h5"] = train_loss_h5
file_record["train_lr_h5"] = train_lr_h5
file_record["time_cost_h5"] = time_cost_h5
file_record["validation_loss_h5"] = validation_loss_h5
file_record.close()


