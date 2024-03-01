import os
from scipy.io import loadmat
import numpy as np
from collections import Counter
import scipy.io as sio

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