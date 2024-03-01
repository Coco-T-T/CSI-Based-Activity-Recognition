#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from scipy.io import loadmat
import numpy as np
from collections import Counter
import scipy.io as sio
import torch

device = torch.device("cuda")

def chk(item):
    s = str(item)
    label = ord(s[4]) - ord('A')
    if s[1] == '1':
        pp = 0
        ll = 0
    elif s[1] == '2':
        pp = 1
        ll = 0
    elif s[1] == '3':
        pp = 2
        ll = 0
    elif s[1] == '4':
        pp = 0
        ll = 0
    elif s[1] == '5':
        pp = 0
        ll = 0
    elif s[1] == '6':
        pp = 0
        ll = 1
    else: 
        pp = 0
        ll = 2
    
    return label, pp, ll


def data_augment(path, act_num, fs, win_tlen, sub_channels, overlap_rate, MorP):
    """
        :param win_tlen: 滑动窗口的时间长度
        :param overlap_rate: 重叠部分比例, [0-100], 百分数:
                             overlap_rate*win_tlen*fs//100 是论文中的重叠量
        :param fs: 原始数据的采样频率
        :param data_iteror: 原始数据的生成器格式
        :return (X, y): X, 切分好的数据， y数据标签
                        X[0].shape == (win_len,)
                        X.shape == (N, win_len)
    """
    overlap_rate = int(overlap_rate)
    # 窗口的长度，单位采样点数
    win_len = int(fs * win_tlen)
    # 重合部分的时间长度，单位采样点数
    overlap_len = int(win_len * overlap_rate / 100)
    # 步长，单位采样点数
    step_len = int(win_len - overlap_len)

    # 滑窗采样增强数据
    X = []
    y = []
    per = []
    loc = []

    ppp = os.listdir(path)

    for item in os.listdir(path):  #文件夹内的数据
        print("Loading -> {}".format(item))
        X_ = []
        y_ = []
        per_ = []
        loc_ = []

        label, pp, ll = chk(item)
        data_path = path + str(item)
        # 数据提取
        D = sio.loadmat(data_path)
        if MorP == 0:
            DD = D['csi_mag']  
        else:
            DD = D['csi_phase']   
        # DD是一个12000*2048的ndarray
        len_data = len(DD[:,1])

        for start_ind, end_ind in zip(range(0, len_data - win_len, step_len),
                                  range(win_len, len_data, step_len)):
            DD_new = DD[start_ind:end_ind][:]
            DD_new = DD_new.T   ###
            X_.append(DD_new)
            y_.append(label)
            per_.append(pp)
            loc_.append(ll)

        X.extend(X_)
        y.extend(y_)
        per.extend(per_)
        loc.extend(loc_)

    X = np.array(X)
    y = np.array(y)
    per = np.array(per)
    loc = np.array(loc)

    return X, y, per, loc


def preprocess(path, act_num, fs, win_tlen, sub_channels,
               overlap_rate, MorP):
    X, y, per, loc = data_augment(path, act_num, fs, win_tlen, sub_channels, overlap_rate, MorP)

    print("-> 数据位置:{}".format(path))
    print("-> 原始数据采样频率:{0}Hz,\n-> 数据增强后共有：{1}条,"
          .format(fs, X.shape[0]))
    print("-> 单个数据长度：{0}采样点,\n-> 重叠量:{1}个采样点,"
          .format(X.shape[1], int(overlap_rate * win_tlen * fs // 100)))
    print("-> 类别数据数目:", sorted(Counter(y).items()))
    return X, y, per, loc


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4 # Bottleneck的输出通道数是输入通道数的4倍
    # Bottleneck有3个卷积层，1*1,3*3,1*1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, shortcut_type='B', num_classes=12):
        
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], shortcut_type, stride=4)
        self.layer4 = self._make_layer(block, 64, layers[3], shortcut_type, stride=4)
        self.avgpool = nn.AvgPool2d((16,6), stride=(16,6))
        # self.fc = nn.Linear(self.feature_num, num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        BN=0
        for layer in self.layer4:
            #print(isinstance(layer, ))
            if isinstance(layer, Bottleneck):
                BN = layer.bn3.weight
                #print(layer)
        # print(x.shape)
        # 得到batch_size行num_classes列的output
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = x
        # print(x.shape)

        x = self.fc(x) 
        return x, feature,BN
        # return F.softmax(x,dim=1) # dim=1，对x每一行的元素进行log_softmax运算


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def exchange_channels(BN1,BN2,size):
    BN1.requires_grad=False
    BN2.requires_grad=False
    for i in range(0,int(size/2)):
        if(BN1[i]<=0):
            BN1[i]=BN2[i]
    for i in range(int(size/2),int(size)):
        if(BN2[i]<=0):
            BN2[i]=BN2[i]
    BN1.requires_grad = True
    BN2.requires_grad = True


from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import random

path = '/home/wutong/code/PE/'
act_num = 12  ## 样本类别 = RSNet.num_classes
feature_num = 256
sub_channels = 2048
fs = 150  
win_tlen = 5  # 4s
win = fs * win_tlen
overlap_rate = 20 
MorP1 = 1  #修改这里可以控制输出的是Mag还是Phase
MorP0 = 0
random_seed = 1
batch_size = 8  ## more than 1
num_epochs = 30
# X: CSI data
# y: action label
# per: person
# loc: location

def LoadData(MorP):
    X, y, per, loc = preprocess(path,   
                      act_num,
                      fs,
                      win_tlen,
                      sub_channels,
                      overlap_rate,
                      MorP
                      )
#print(X.shape)  # 单条数据维度为750*2048
#print(len(X))
#X=X.reshape 
    all_data = torch.from_numpy(X)
    all_label = torch.from_numpy(y)
    all_dataset = TensorDataset(all_data, all_label)
    ## 80%用于训练
    train_size = int(len(all_dataset) * 0.8)
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1))

    all_data_loader = DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return all_data_loader,train_data_loader,test_data_loader,all_label

all_data_loader1,train_data_loader1,test_data_loader1,all_label = LoadData(MorP0)
all_data_loader2,train_data_loader2,test_data_loader2,all_label = LoadData(MorP1)

net1 = resnet50()
net1 = net1.cuda()
net2 = resnet50()
net2 = net2.cuda()

Loss_list1 = []
Accuracy_list1 = []
acc1 = []

Loss_list2 = []
Accuracy_list2 = []
acc2 = []

optimizer1 = optim.SGD(net1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
loss_function = nn.CrossEntropyLoss()  ## 交叉熵损失函数


for epoch in range(num_epochs):
    net1.train()
    sum_loss1 = 0.0  # 损失数量
    correct1 = 0.0  # 准确数量
    total1 = 0.0  # 总共数量
    sum_loss2 = 0.0  # 损失数量
    correct2 = 0.0  # 准确数量
    total2 = 0.0  # 总共数量

    for i,(X,y) in enumerate(train_data_loader1):
        length = len(train_data_loader1)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        # X = X.transpose(1,2)
        
        optimizer1.zero_grad()
        # outputs,fea = net(X)  ### change
        X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen) # batch_size, channels, row, column
        X = X.to(device)
        outputs1, features1, BN1  = net1(X) # outputs 5*12, features 5*2967552
        # print(outputs.shape)
        # print(features.shape)
        # outputs_all = torch.cat(outputs_all, outputs, dim=0)
        
        loss1 = loss_function(outputs1, y)
        loss1.backward()
        optimizer1.step()
        
        sum_loss1 += loss1.item()
        _, predicted1 = torch.max(outputs1.data, 1)
        total1 += y.size(0)
        correct1 += (predicted1 == y).sum().item()

        print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% ' 
            % (epoch + 1, (i + 1), length, sum_loss1 / (i + 1), 100. * correct1 / total1))
    
    Loss_list1.append(sum_loss1 / (len(train_data_loader1)))
    Accuracy_list1.append(correct1 / total1)
    
    net2.train()
    for i,(X,y) in enumerate(train_data_loader2):
        length = len(train_data_loader2)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        # X = X.transpose(1,2)

        optimizer2.zero_grad()
        # outputs,fea = net(X)  ### change
        X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen) # batch_size, channels, row, column
        X = X.to(device)
        outputs2, features2, BN2  = net2(X)
        # print(outputs.shape)
        # print(features.shape)
        # outputs_all = torch.cat(outputs_all, outputs, dim=0)
            
        loss2 = loss_function(outputs2, y)
        loss2.backward()
        optimizer2.step()

        sum_loss2 += loss2.item()
        _, predicted2 = torch.max(outputs2.data, 1)
        total2 += y.size(0)
        correct2 += (predicted2 == y).sum().item()
        # correct += predicted.eq(y.data).cpu().sum()

        print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% ' 
            % (epoch + 1, (i + 1), length, sum_loss2 / (i + 1), 100. * correct2 / total2))
    
    
    Loss_list2.append(sum_loss2 / (len(train_data_loader2)))
    Accuracy_list2.append(correct2 / total2)
    
    # exchange_channels(BN1,BN2,256)
    
    print("Waiting Test!")
    with torch.no_grad():  # 没有求导
        correct1 = 0.0
        total1 = 0.0
        for test_i,(test_X,test_y) in enumerate(test_data_loader1):
            net1.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            X = test_X.type(torch.FloatTensor)  
            y = test_y.type(torch.LongTensor)
            y = y.to(device)
            # X = X.transpose(1,2)
            # outputs,fea = net(X)  ### change
            X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen)
            X = X.to(device)

            outputs1, features1,_ = net1(X)
            # 取得分最高的那个类 (outputs.data的索引号)
            
            _, predicted1 = torch.max(outputs1.data, 1)
            total1 += y.size(0)
            correct1 += (predicted1 == y).sum().item()
            # correct += predicted.eq(y.data).cpu().sum()
            # if test_i == 100:
            #     break
        print('网络1测试分类准确率为：%.3f%%' % (100. * correct1 / total1))
        acc1.append( 100. * correct1 / total1)
    with torch.no_grad():  # 没有求导
        correct2 = 0.0
        total2 = 0.0
        for test_i,(test_X,test_y) in enumerate(test_data_loader2):
            net2.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            X = test_X.type(torch.FloatTensor)  
            y = test_y.type(torch.LongTensor)
            y = y.to(device)
            # X = X.transpose(1,2)
            # outputs,fea = net(X)  ### change
            X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen)
            X = X.to(device)

            outputs2, features2,_ = net2(X)
            
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted2 = torch.max(outputs2.data, 1)
            total2 += y.size(0)
            correct2 += (predicted2 == y).sum().item()
            # correct += predicted.eq(y.data).cpu().sum()
            # if test_i == 100:
            #     break
        print('网络2测试分类准确率为：%.3f%%' % (100. * correct2 / total2))
        acc2.append( 100. * correct2 / total2)
    
print("网络1 loss:")
print(Loss_list1)

print("网络1 train_acc:")
print(Accuracy_list1)

print("网络1 test_acc:")
print(acc1)

print("网络2 loss:")
print(Loss_list2)

print("网络1 train_acc:")
print(Accuracy_list2)

print("网络2 test_acc:")
print(acc2)

print("Training Finished, TotalEPOCH=%d" % num_epochs)


features_all1 = []
features_all2 = []
    
label_all1 = []
label_all2 = []
    
with torch.no_grad():  # 没有求导
    for i,(X,y) in enumerate(all_data_loader1):
        net1.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
        X = X.type(torch.FloatTensor)  
        y = y.type(torch.LongTensor)
        y = y.to(device)

        X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen)
        X = X.to(device)

        outputs1, features1,_ = net1(X)
        # 取得分最高的那个类 (outputs.data的索引号)
            
        features_all1.append(features1)
        label_all1.append(y)
            
with torch.no_grad():  # 没有求导
    for i,(X,y) in enumerate(all_data_loader2):
        net2.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
        X = X.type(torch.FloatTensor)  
        y = y.type(torch.LongTensor)
        y = y.to(device)
        
        X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen)
        X = X.to(device)

        outputs2, features2,_ = net2(X)
            
        features_all2.append(features2)
        label_all2.append(y)
         

save_pt_path_1 = "/home/wutong/code/mag_resnet50.pt"
save_pt_path_2 = "/home/wutong/code/phase_resnet50.pt"

torch.save(net1.state_dict(), save_pt_path_1)
torch.save(net2.state_dict(), save_pt_path_2)


all_features1 = torch.cat(features_all1, dim=0)
all_features2 = torch.cat(features_all2, dim=0)
import numpy as np
from scipy.io import savemat
features = torch.cat((all_features1, all_features2), dim=1)
#features = torch.cat((features, all_label), dim=0)
#features_np = features.cpu().numpy()

# 保存为 .mat 文件
#savemat('dataToClassfy.mat', {'features': features_np})


import torch
import numpy
from scipy.io import savemat

# 假设 feature 是一个 PyTorch 张量
# 例如：feature = torch.randn(100, 10)  # 一个 100x10 的特征张量

# 确保张量在 CPU 上
features = features.cpu()

# 将 PyTorch 张量转换为 NumPy 数组
feature_np = features.detach().numpy()

# 保存为 .mat 文件
savemat('/home/wutong/code/features.mat', {'features': feature_np})

labels = torch.cat(label_all1,dim=0)
labels = labels.cpu()
labels  = labels.detach().numpy()

savemat('/home/wutong/code/labels.mat', {'labels': labels})