import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import dill

from resnet import resnet50
from data_process import preprocess

path = '/home/wutong/code/PE/'
act_num = 12  ## 样本类别 = RSNet.num_classes
feature_num = 256
sub_channels = 2048
fs = 150  
win_tlen = 5  # 4s
win = fs * win_tlen
overlap_rate = 20 
MorP = 1  #修改这里可以控制输出的是Mag还是Phase
random_seed = 1
batch_size = 8  ## more than 1
num_epochs = 30
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

net = resnet50()
net = net.cuda()

device = torch.device("cuda")

Loss_list = []
Accuracy_list = []
acc = []

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
loss_function = nn.CrossEntropyLoss()  ## 交叉熵损失函数
for epoch in range(num_epochs):
    net.train()
    sum_loss = 0.0  # 损失数量
    correct = 0.0  # 准确数量
    total = 0.0  # 总共数量
    for i,(X,y) in enumerate(train_data_loader):
        length = len(train_data_loader)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        # X = X.transpose(1,2)

        optimizer.zero_grad()
        # outputs,fea = net(X)  ### change
        X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen) # batch_size, channels, row, column
        X = X.to(device)
        outputs, features = net(X) # outputs 5*12, features 5*2967552

        # print(outputs.shape)
        # print(features.shape)
        # outputs_all = torch.cat(outputs_all, outputs, dim=0)
        # features_all = torch.cat(features_all, features, dim=0)

        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        # correct += predicted.eq(y.data).cpu().sum()

        print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% ' 
            % (epoch + 1, (i + 1), length, sum_loss / (i + 1), 100. * correct / total))

    Loss_list.append(sum_loss / (len(train_data_loader)))
    Accuracy_list.append(correct / total)

    print("Waiting Test!")
    with torch.no_grad():  # 没有求导
        correct = 0.0
        total = 0.0
        for test_i,(test_X,test_y) in enumerate(test_data_loader):
            net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            X = test_X.type(torch.FloatTensor)  
            y = test_y.type(torch.LongTensor)
            y = y.to(device) 
            # X = X.transpose(1,2)
            # outputs,fea = net(X)  ### change
            X = X.reshape(batch_size, 1, sub_channels, fs*win_tlen)
            X = X.to(device)

            outputs, features = net(X)   
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # correct += predicted.eq(y.data).cpu().sum()
            # if test_i == 100:
            #     break
        print('测试分类准确率为：%.3f%%' % (100. * correct / total))
        acc.append( 100. * correct / total)

print("loss:")
print(Loss_list)

print("train_acc:")
print(Accuracy_list)

print("test_acc:")
print(acc)
# x1 = range(0, num_epochs)
# x2 = range(0, num_epochs)
# x3 = range(0, num_epochs)
# y1 = Accuracy_list
# y2 = Loss_list
# y3 = acc
# plt.subplot(3, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Train accuracy vs. epoches')
# plt.ylabel('Train accuracy')
# plt.subplot(3, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.title('Train loss vs. epoches')
# plt.ylabel('Train loss')
# plt.subplot(3,1,3)
# plt.plot(x3, y3, 'o-')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.show()

print("Training Finished, TotalEPOCH=%d" % num_epochs)

nm = 'phase_model_256'
save_pt_path = '/home/wutong/code/train/' + nm + "/resnet50.pt"
save_test_path = '/home/wutong/code/train/' + nm + "/dataloader_test.pkl"
save_train_path = '/home/wutong/code/train/' + nm + "/dataloader_train.pkl"
save_all_path = '/home/wutong/code/train/' + nm + "/dataloader_all.pkl"

torch.save(net.state_dict(), save_pt_path)

with open(save_test_path,'wb') as f:
    dill.dump(test_data_loader, f)

with open(save_train_path,'wb') as f:
    dill.dump(train_data_loader, f)

with open(save_all_path,'wb') as f:
    dill.dump(all_data_loader, f)

# fea_path = "./feature.csv"
# fea_dim = 32
# label_all = []
# feature_all = []

# for i,(X,y) in enumerate(all_data_loader):
#     net.eval()  
#     X = X.type(torch.FloatTensor)
#     y = y.type(torch.LongTensor)
#     outputs,fea = net(X)  
#     fea = fea.squeeze()
#     for i in range(batch_size):
#         label_all.append(y[i])
#         feature_all.append(fea[i])

# cnt = 0
# with open(fea_path, "w") as outputfile:
#     outputfile.write("label")  #输出表格名
#     for i in range(fea_dim):
#         outputfile.write(","+str(i))
#     outputfile.write("\n")
#     for i in label_all :  #输出label和特征
#         outputfile.write(f"{i}")
#         for j in feature_all[cnt] :
#             outputfile.write(f",{j}")
#         outputfile.write("\n")
#         cnt += 1