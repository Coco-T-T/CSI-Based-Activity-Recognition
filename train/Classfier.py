#!/usr/bin/env python
# coding: utf-8

import torch
import os
from scipy.io import loadmat
import numpy as np
from collections import Counter
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


features_mat = loadmat('/home/wutong/code/features.mat')
labels_mat = loadmat('/home/wutong/code/labels.mat')

features = features_mat['features']  # 替换为实际的变量名
labels = labels_mat['labels']        # 替换为实际的变量名
labels = labels[0]
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)  # 假设标签是整数
labels_tensor = labels_tensor.t()
print(len(features))
print(len(labels))

device = torch.device("cuda:0")
features_tensor = features_tensor.to(device)
labels_tensor = labels_tensor.to(device)


all_dataset = TensorDataset(features_tensor, labels_tensor)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(512, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
def train(train_loader, model, criterion, optimizer, epochs):
    #model.train()
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

def test(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


import torch.optim as optim
model = SimpleMLP().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_samples=680
# 4. 初始化训练和测试 DataLoader
batch_size = 8
# 假设我们使用80%的数据进行训练，20%的数据进行测试
train_size = int(num_samples * 0.8)
test_size = num_samples - train_size

train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 5. 训练模型
epochs = 30
train(train_loader, model, criterion, optimizer, epochs)

# 6. 测试模型性能
accuracy = test(test_loader, model)
print(f'Test Accuracy: {accuracy:.2f}%')