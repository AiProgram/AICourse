from scipy.io import loadmat
import h5py
import numpy as np
import random
import os
from sklearn.utils import shuffle

def Normalization(x): # Min-Max 归一化到[-1,1]
    for row in x[1:]:
        min = np.min(row)
        max=np.max(row)
        for i in range(len(row)):
            row[i]=2*(row[i]-min)/(max-min)-1

def get_our_data():
    #获取本文件所在的目录
    thisFolder=os.path.dirname(os.path.abspath(__file__))
    #读取npy格式的数据文件
    #手写体数据
    data=np.load(thisFolder+"/number_data.npy")
    Normalization(data)
    #对应标签
    label=np.load(thisFolder+"/label_data.npy")

    #使用标签数组的(0,0)位置储存当前收集的数据条数
    n_data = int(label[0][0])
    #分割训练集核测试集
    train_data_len = int(n_data*0.1)#设定分割点
    #因为收集时是按手写数字同时收集多个，所以需要打乱数据，防止分割的大部分是同一数字
    data,label=shuffle(data[1:n_data],label[1:n_data])
    return data[train_data_len:n_data], data[:train_data_len], label[train_data_len:n_data], label[:train_data_len]
    # 返回的label是one-hot编码