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
DATASET = 'datasets/' + sys.argv[1] 
GROUND_TRUTH = 'pm10'
SENSOR = sys.argv[2]
MODEL = sys.argv[3]
CROSS_VALID = 10
try:
    CROSS_VALID = int(sys.argv[4])
except:
    CROSS_VALID = 10
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
    
home = pd.read_csv(os.path.join(ROOTS, DATASET))[[GROUND_TRUTH, SENSOR]]
windowDataset = WindowDataset(home[[SENSOR]].to_numpy(),
                              home[[GROUND_TRUTH]].to_numpy())

print('Done.')



trainDataLoaders = []
validDataLoaders = []
testDataLoaders = []

for train_index, valid_index, test_index in time_series_split(len(windowDataset.X), CROSS_VALID):
    trainDataLoader, validDataLoader, testDataLoader = generate_data_loader(windowDataset, train_index, valid_index, test_index)
    trainDataLoaders.append(trainDataLoader)
    validDataLoaders.append(validDataLoader)
    testDataLoaders.append(testDataLoader)

    
for i, trainDataLoader, validDataLoader, testDataLoader in zip(range(CROSS_VALID), trainDataLoaders, validDataLoaders, testDataLoaders):
    print('\n\n\n##################')
    print('#', i + 1, ': ')


    # Train indices
    train_indices = trainDataLoader.dataset.indices
    valid_indices = validDataLoader.dataset.indices
    test_indices = testDataLoader.dataset.indices
    validX = validDataLoader.dataset.dataset[valid_indices][0]
    validy = validDataLoader.dataset.dataset[valid_indices][1]
    testX = testDataLoader.dataset.dataset[test_indices][0]
    testy = testDataLoader.dataset.dataset[test_indices][1]
    original = np.array(home[SENSOR])


    # Model definition
    data_dim = 1
    hidden_dim = 16
    output_dim = 1 
    learning_rate = 0.02
    nb_epochs = 500
    class_dim = 2
    uModel = UModel(data_dim, hidden_dim, class_dim, output_dim, 1).to(device)

    
    
    print('Step 1. Single model training...')
    model, train_hist = model_training(uModel, trainDataLoader, validDataLoader, mode=0)
    mae, pred = model_predict(uModel, testX, testy, mode=0, class_value=1)
    linear_mae, linear_pred = model_predict(uModel, testX, testy, mode=0, class_value=-1)

    print('Done.')
    print('component MAE SCORE : ', mae)
    print('linear MAE SCORE : ', linear_mae)


    
    print('Step 2. Generate classifier datasets...')
    trainClassifyDataLoader, validClassifyDataLoader, testClassifyDataLoader = generate_classify_data_loader(model, 
                                                                                        trainDataLoader,
                                                                                        validDataLoader,
                                                                                        testDataLoader, 
                                                                                        original)
    model, train_hist = model_training(uModel, trainClassifyDataLoader, validClassifyDataLoader, mode=1)
    classTestX = testClassifyDataLoader.dataset.dataset[test_indices][0]
    classTesty = testClassifyDataLoader.dataset.dataset[test_indices][1]
    class_mae, class_pred = model_predict(uModel, classTestX, classTesty, mode=1)
    print('Done.')

    

    print('Step 3. Unified model training...')
    model, train_hist = model_training(uModel, trainDataLoader, validDataLoader, mode=-1)
    torch.save(uModel.state_dict(), 'train_models/' + MODEL + '_' + sys.argv[1].split('-')[0].split('.')[0] + '_' + SENSOR + '_' + str(i+1) + '.pkl')

    unified_mae, unified_pred = model_predict(uModel, testX, testy, mode=-1)
    print('Done.')
    print('unified MAE SCORE : ', unified_mae)