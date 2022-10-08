##
## WindowDataset.py
## 시계열 처리를 위한 데이터 분할
##
## ┌────────────────────────────────────────────────────────────┐
## │                           Data(X)                          │
## ├────────────────────────────────────────────────────────────┤
## │                           Data(y)                          │
## └────────────────────────────────────────────────────────────┘
## ┌────────┐
## │   iw   │
## └─┬──────┤
##   │  ow  │
##   └──────┘
##      ┌────────┐
##      │   iw   │
##      └─┬──────┤
##        │  ow  │
##        └──────┘
##           ┌────────┐
##           │   iw   │    
##           └─┬──────┤ ...
##             │  ow  │
##             └──────┘
##

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
from Models import model_predict

class WindowDataset(Dataset):
    def __init__(self, 
                 Xdatas :np.ndarray, # valid shape: (ndata,) or (ndata, feature)
                 ydatas :np.ndarray, # valid shape: (ndata,) or (ndata, feature)
                 input_window=20,
                 output_window=1,
                 stride=1):
        
        assert input_window >= output_window, 'input window size must be less then output window size(input_window >= output_window).'
        assert len(Xdatas.shape) < 3, 'source(X) and target(y) data must be two-dimensional or less data.'
        assert Xdatas.shape[0] == ydatas.shape[0], 'source(X) and target(y) data must be same shape.'
        
        ndatas = len(Xdatas)
        nsamples = (ndatas - input_window) // stride + 1
        
        nfeatures = 1 if len(Xdatas.shape) == 1 else Xdatas.shape[1]
        noutputs = 1 if len(ydatas.shape) == 1 else ydatas.shape[1]

        X = np.zeros([nsamples, input_window, nfeatures], dtype=np.float32)
        y = np.zeros([nsamples, output_window, noutputs], dtype=np.float32)
        
        for i in range(nsamples):
            base_idx = stride * i + input_window
            X[i, :] = Xdatas[base_idx-input_window:base_idx]
            y[i, :] = ydatas[base_idx-output_window:base_idx]
        
        # 중복된 차원 제거
        self.X = X.squeeze(-2) if X.shape[-2] == 1 else X
        self.y = y.squeeze(-2) if y.shape[-2] == 1 else y
        
        
        # TODO:
        # 여러개의 feature을 받을 시에 shape의 형태가
        # (ndata, window, feature)인지, (ndata, feature, window)인지 확인하고
        # 그에 맞게 코드를 수정
        #
        #
            
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.X)
    
    

def generate_data_loader(windowDataset, train_index, valid_index, test_index, batch_size=400):
    
    def train_test_split(dataset, train_index, valid_index, test_index):
        trainDataset = Subset(dataset, np.arange(len(dataset))[train_index])
        validDataset = Subset(dataset, np.arange(len(dataset))[valid_index])
        testDataset  = Subset(dataset, np.arange(len(dataset))[test_index])
        return trainDataset, validDataset, testDataset
    
    trainWindowDataset, validWindowDataset, testWindowDataset = train_test_split(windowDataset, train_index, valid_index, test_index)
    trainDataLoader = DataLoader(trainWindowDataset, batch_size=400)
    validDataLoader = DataLoader(validWindowDataset, batch_size=400)
    testDataLoader = DataLoader(testWindowDataset, batch_size=400)
        
    return trainDataLoader, validDataLoader, testDataLoader


def generate_classify_data_loader(model, trainDataLoader, validDataLoader, testDataLoader, original):
    train_indices = trainDataLoader.dataset.indices
    valid_indices = validDataLoader.dataset.indices
    test_indices = testDataLoader.dataset.indices
    indices = np.concatenate([train_indices, valid_indices, test_indices])
    
    X = testDataLoader.dataset.dataset[indices][0]
    y = testDataLoader.dataset.dataset[indices][1]
    _, pred = model_predict(model, X, y, mode=0, class_value=1)
    _, linear_pred = model_predict(model, X, y, mode=0, class_value=-1)
    
    margin = -1
    pad_window = 20
    smooth_window = 100
    
    def pad_left(arr, value, pad_length=pad_window-1):
        return np.pad(arr, (pad_length, 0), 'constant', constant_values=value)
    
    # generate classifier dataset
    dataset = np.abs(y.ravel() - linear_pred) - np.abs(y.ravel() - pred) + margin
    dataset = np.where(dataset > 0, 1, 0)
    dataset = pad_left(dataset, dataset[0])
    
    # smoothing
    smooth_dataset = np.convolve(dataset, np.ones(smooth_window) / smooth_window, 'same')
    smooth_dataset = np.where(smooth_dataset > 0.5, 1, 0)
    
    # stability
    stable_dataset = np.bitwise_xor(dataset, np.roll(dataset, 1))
    stable_dataset = np.convolve(stable_dataset, np.ones(smooth_window), 'same')
    
    # min_max_scale & magnify
    classify_dataset = (lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-4))(smooth_dataset + stable_dataset)
    classify_dataset = (lambda x, n: ((n + 1) * x) / (n*x + 1))(classify_dataset, 3)
    
    X = pad_left(original[indices], original[indices][0]).reshape(-1, 1)
    y = classify_dataset.reshape(-1, 1)
    
    return generate_data_loader(WindowDataset(X, y), train_indices, valid_indices, test_indices)