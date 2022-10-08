import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from WindowDataset import WindowDataset
from WindowDataset import generate_data_loader
from WindowDataset import generate_classify_data_loader

from Models_eval import check_inference_time


############
# 이 부분을 수정하세요.
from Models_eval import UnifiedLSTMModel # 모델에 맞게 불러오기
PATH = 'lstm_home1_sensor1.pkl' # 모델이름(경로)


ROOTS = '' # 수정 x
FILENAME = ['home1.csv'] # home1, 2, 3에 맞게 수정
GROUND_TRUTH = ['pm10']  # 수정 x
FEATURES = ['sensor1']   # sensor1, 2, 3, 4에 맞게 수정
############


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()




homes = [pd.read_csv(os.path.join(ROOTS, file_name))[GROUND_TRUTH + FEATURES] for file_name in FILENAME]
windowDatasets = [WindowDataset(home[[feature]].to_numpy(), 
                                home[GROUND_TRUTH].to_numpy())
                  for home in homes
                  for feature in FEATURES]


def timeSeriesSplit(data_len, n_splits=5):
    indices = [i * (len(windowDataset.X) // (n_splits + 2)) + len(windowDataset.X) % (n_splits + 2) for i in range(1, n_splits + 3)]
    return ((np.arange(0, indices[i]),
             np.arange(indices[i], indices[i + 1]),
             np.arange(indices[i + 1], indices[i + 2]))
            for i in range(n_splits))



linear_time_list = []
component_time_list = []
unified_time_list = []
for nWD, windowDataset in zip(range(len(windowDatasets)), windowDatasets):
    
    print('\n\n\n##################')
    print('#', nWD + 1, 'windowDataset')
    print('Dataset: ', nWD // len(FEATURES) + 1)
    print('Sensor: ', nWD % len(FEATURES) + 1)
    print('##################')


    testDataLoaders = []

    for train_index, valid_index, test_index in timeSeriesSplit(len(windowDataset.X), 1):
        _, _, testDataLoader = generate_data_loader(windowDataset, train_index, valid_index, test_index)
        testDataLoaders.append(testDataLoader)
        
    for i, testDataLoader in zip(range(10), testDataLoaders):
        print('\n\n\n##################')
        print('#', i + 1, ': ')


        # Train indices
        test_indices = testDataLoader.dataset.indices
        testX = testDataLoader.dataset.dataset[test_indices][0]
        testy = testDataLoader.dataset.dataset[test_indices][1]


        # Model definition
        device = 'cpu'
        uModel = UnifiedLSTMModel(1, 16, 2, 1, 1).to(device)
        uModel.load_state_dict(torch.load(PATH))

        
        # Check inference time
        print("Unified: ")
        unified_time = check_inference_time(uModel, -1, torch.Tensor(testX), testy)

        print("Single(Component): ")
        component_time = check_inference_time(uModel, 0, torch.Tensor(testX), testy)
        
        print("Single(Linear): ")
        linear_time = check_inference_time(uModel, 1, torch.Tensor(testX), testy)


        # results
        linear_time_list.append(linear_time)
        component_time_list.append(component_time)
        unified_time_list.append(unified_time)
