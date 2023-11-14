from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

#from rsnet import rsnet34
from data_process import preprocess

path = './PE/'
act_num = 12  ## 样本类别 = RSNet.num_classes
sub_channels = 2048
fs = 150  
win_tlen = 5  # 4s
win = fs * win_tlen
overlap_rate = 20 
MorP = 0  #修改这里可以控制输出的是Mag还是Phase
random_seed = 1
batch_size = 5  ## more than 1
num_epochs = 10
# X: CSI data
# y: action label
# per: person
# loc: location
X, y, per, loc = preprocess(path,   
                  act_num,
                  fs,
                  win_tlen,
                  sub_channels,
                  overlap_rate,
                  MorP
                  )
print(X.shape)  # 单条数据维度为750*2048
print(len(X))
#X=X.reshape 
all_data = torch.from_numpy(X)
all_label = torch.from_numpy(y)
all_dataset = TensorDataset(all_data, all_label)

## 80%用于训练
train_size = int(len(all_dataset) * 0.8)
test_size = len(all_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])

all_data_loader = DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# with open('./dataloader_test.pkl','wb') as f:
#     dill.dump(test_data_loader, f)

# with open('./dataloader_train.pkl','wb') as f:
#     dill.dump(train_data_loader, f)

# with open('./dataloader_all.pkl','wb') as f:
#     dill.dump(all_data_loader, f)