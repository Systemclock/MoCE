import torch
from sklearn import preprocessing
import scipy.io as sio
import numpy as np

import warnings

from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

path = '../dataset/'

class HW(Dataset):

    def __init__(self, path):
        data = sio.loadmat(path + 'handwritten_6viewsy.mat')
        x1 = data['X'][0][0]
        x2 = data['X'][0][1]
        x3 = data['X'][0][2]
        x4 = data['X'][0][3]
        x5 = data['X'][0][4]
        x6 = data['X'][0][5]
        min_max_scaler = preprocessing.MinMaxScaler()
        self.x1 = min_max_scaler.fit_transform(x1).astype(np.float32)
        self.x2 = min_max_scaler.fit_transform(x2).astype(np.float32)
        self.x3 = min_max_scaler.fit_transform(x3).astype(np.float32)
        self.x4 = min_max_scaler.fit_transform(x4).astype(np.float32)
        self.x5 = min_max_scaler.fit_transform(x5).astype(np.float32)
        self.x6 = min_max_scaler.fit_transform(x6).astype(np.float32)
        self.x = [torch.from_numpy(self.x1), torch.from_numpy(self.x2), torch.from_numpy(self.x3), torch.from_numpy(self.x4),
                  torch.from_numpy(self.x5), torch.from_numpy(self.x6)]
        Y = data['Y'].flatten().astype(np.int32)
        if np.min(Y) == 1:
            Y = Y - 1
        self.labels = torch.from_numpy(Y)
        self.views = 6

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]), torch.from_numpy(self.x3[idx]), torch.from_numpy(
            self.x4[idx]),torch.from_numpy(self.x5[idx]), torch.from_numpy(
            self.x6[idx])]
        y = self.labels[idx]
        index = torch.from_numpy((np.array(idx)))
        return x, y, index


class BDGP(Dataset):

    def __init__(self, path):
        data = sio.loadmat(path + 'BDGP.mat')
        x1 = data['X'][0][0]
        x2 = data['X'][0][1]
        min_max_scaler = preprocessing.MinMaxScaler()
        self.x1 = min_max_scaler.fit_transform(x1).astype(np.float32)
        self.x2 = min_max_scaler.fit_transform(x2).astype(np.float32)
        self.x = [torch.from_numpy(self.x1), torch.from_numpy(self.x2)]
        Y = data['Y'].flatten().astype(np.int32)
        if np.min(Y) == 1:
            Y = Y - 1
        self.labels = torch.from_numpy(Y)
        self.views = 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])]
        y = self.labels[idx]
        index = torch.from_numpy((np.array(idx)))
        return x, y, index



def load_data(dataset):
    print("load: ", dataset)
    if dataset == 'BDGP':
        return BDGP(path)
    elif dataset == "HW":
        return HW(path)
    else:
        raise ValueError('Not defined for loading %s' % dataset)
