from scipy.io import loadmat
import h5py
import numpy as np
import random
import os
from sklearn.utils import shuffle

def Normalization(x):
    for row in x[1:]:
        min=np.min(row)
        max=np.max(row)
        for i in range(len(row)):
            row[i]=2*(row[i]-min)/(max-min)-1

def get_our_data():
    thisFolder=os.path.dirname(os.path.abspath(__file__))
    data=np.load(thisFolder+"/number_data.npy")
    Normalization(data)
    label=np.load(thisFolder+"/label_data.npy")
    n_data = int(label[0][0])
    train_data_len = int(n_data*0.1)#设定分割点
    data,label=shuffle(data[1:n_data],label[1:n_data])
    return data[train_data_len:n_data], data[:train_data_len],label[train_data_len:n_data], label[:train_data_len]