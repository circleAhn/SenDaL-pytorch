import sys
import os
import numpy as np
import pandas as pd
import torch
from WindowDataset import WindowDataset
from WindowDataset import generate_data_loader
from WindowDataset import generate_classify_data_loader
from WindowDataset import time_series_split

from Models import model_training
from Models import model_predict



ROOTS = ''
DATASET = ['datasets/home1.csv', 'datasets/home2.csv', 'datasets/home3.csv']
GROUND_TRUTH = ['pm10']
SENSOR = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
MODEL = sys.argv[1]
CROSS_VALID = 10
try:
    CROSS_VALID = int(sys.argv[2])
except:
    CROSS_VALID = 10
    
TYPE = sys.argv[3]
assert TYPE == '1' or TYPE == '2', 'Type must be 1(single) or 2(unified)'
if TYPE == '1':
    TYPE = 'single'
else:
    TYPE = 'unified'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
if MODEL.lower() == 'lstm':
    from Models import UnifiedLSTMModel as UModel
elif MODEL.lower() == 'gru':
    from Models import UnifiedGRUModel as UModel
elif MODEL.lower() == 'plstm' or MODEL.lower() == 'phasedlstm':
    from Models import UnifiedPhasedLSTMModel as UModel
elif MODEL.lower() == 'trans' or MODEL.lower() == 'transformer':
    from Models import UnifiedTransformerModel as UModel
else:
    raise Exception('Invalid model type')


    
print('Generating window datasets...')
    
homes = [pd.read_csv(os.path.join(ROOTS, file_name))[GROUND_TRUTH + SENSOR] for file_name in DATASET]
windowDatasets = [WindowDataset(home[[feature]].to_numpy(), 
                                home[GROUND_TRUTH].to_numpy())
                  for home in homes
                  for feature in SENSOR]

print('Done.')



rmse_list = []

for nWD, windowDataset in zip(range(len(windowDatasets)), windowDatasets):
    
    print('\n\n\n##################')
    print('#', nWD + 1, 'windowDataset')
    print('Dataset: ', nWD // len(SENSOR) + 1)
    print('Sensor: ', nWD % len(SENSOR) + 1)
    print('##################')

    testDataLoaders = []

    for train_index, valid_index, test_index in time_series_split(len(windowDataset.X), CROSS_VALID):
        _, _, testDataLoader = generate_data_loader(windowDataset, train_index, valid_index, test_index)
        testDataLoaders.append(testDataLoader)
    
    for i, testDataLoader in zip(range(CROSS_VALID), testDataLoaders):
        #print('\n\n\n##################')
        


        # Train indices
        test_indices = testDataLoader.dataset.indices
        testX = testDataLoader.dataset.dataset[test_indices][0]
        testy = testDataLoader.dataset.dataset[test_indices][1]


        # Model definition
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = 'cpu'
        data_dim = 1
        hidden_dim = 16
        output_dim = 1 
        class_dim = 2
        PATH = 'train_models/' + MODEL + '_home' + str(nWD // len(SENSOR) + 1) + '_sensor' + str(nWD % len(SENSOR) + 1) + '_' + str(i + 1) + 'of' + str(CROSS_VALID) + '_' + TYPE + '.pkl'
        
        uModel = UModel(data_dim, hidden_dim, class_dim, output_dim, 1).to(device)
        uModel.load_state_dict(torch.load(PATH, map_location=device))
        
        if TYPE == 'single':
            mae, _ = model_predict(uModel, testX, testy, mode=0, class_value=1)
        else:
            mae, _ = model_predict(uModel, testX, testy, mode=-1)

        rmse_list.append(mae)
        
        print('#', i + 1, ': Done.')
        

print('Home1: ', np.sqrt(np.mean(rmse_list[:40])))
print('Home2: ', np.sqrt(np.mean(rmse_list[40:80])))
print('Home3: ', np.sqrt(np.mean(rmse_list[80:])))
print('Overall: ', np.sqrt(np.mean(rmse_list)))
        
  
