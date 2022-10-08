##
## Models_eval.py
## 시계열 처리를 위한 모델
##

import torch
from torch import nn
from PhasedLSTM import PhasedLSTM
from TransformerEncoder import TransformerEncoder
import numpy as np
from time import time
device = 'cpu'

    
class UnifiedLSTMModel(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 class_dim, 
                 output_dim, 
                 nlayers,
                 linear_period=12*5,
                 comp_period=12):
        super(UnifiedLSTMModel, self).__init__()
        
        self.class_dim = class_dim
        self.training_mode = -1
        self.class_value = torch.Tensor([-1]).to(device)
        self.test_mode = 0
        self.tic = 0
        self.classifier_weight = 0
        self.linear_period = linear_period
        self.comp_period = comp_period
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)
        
        
        self.classifier = nn.Linear(input_dim, output_dim)
        
        self.comp_in = nn.LSTM(input_dim, hidden_dim, num_layers=nlayers, batch_first=True)
        self.comp_out = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        
        def simple(x):
            x = self.fc_in(x)
            x = self.relu(x)
            x = self.fc_out(x)
            return x
        
        def component(x):
            x, _ = self.comp_in(x)
            x = self.comp_out(x)
            return x
        
        def classifier(x):
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x
        
        # single model training
        if self.training_mode == 0:
            x = component(x) if self.class_value > 0.5 else simple(x)
        
        # classifer training
        elif self.training_mode == 1:
            x = classifier(x)

        # unified model training
        else:
            if isinstance(self.classifier_weight, int):
                self.classifier_weight = classifier(x)
            
            if self.test_mode == 1:
                if self.classifier_weight.mean() > 0.5:
                    if self.tic % self.comp_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = component(x)
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = simple(x)
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
                x = torch.where(self.classifier_weight.mean() > 0.5, component(x), simple(x))
        
        return x[:, -1, :]
    
    
    
    
class UnifiedGRUModel(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 class_dim, 
                 output_dim, 
                 nlayers,
                 linear_period=12*5,
                 comp_period=12):
        super(UnifiedGRUModel, self).__init__()
        
        self.class_dim = class_dim
        self.training_mode = -1
        self.class_value = torch.Tensor([-1]).to(device)
        self.test_mode = 0
        self.tic = 0
        self.classifier_weight = 0
        self.linear_period = linear_period
        self.comp_period = comp_period
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)
        
        
        self.classifier = nn.Linear(input_dim, output_dim)
        
        self.comp_in = nn.GRU(input_dim, hidden_dim, num_layers=nlayers, batch_first=True)
        self.comp_out = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        
        def simple(x):
            x = self.fc_in(x)
            x = self.relu(x)
            x = self.fc_out(x)
            return x
        
        def component(x):
            x, _ = self.comp_in(x)
            x = self.comp_out(x)
            return x
        
        def classifier(x):
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x
        
        # single model training
        if self.training_mode == 0:
            x = component(x) if self.class_value > 0.5 else simple(x)
        
        # classifer training
        elif self.training_mode == 1:
            x = classifier(x)

        # unified model training
        else:
            if isinstance(self.classifier_weight, int):
                self.classifier_weight = classifier(x)
            
            if self.test_mode == 1:
                if self.classifier_weight.mean() > 0.5:
                    if self.tic % self.comp_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = component(x)
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = simple(x)
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
                x = torch.where(self.classifier_weight.mean() > 0.5, component(x), simple(x))
        
        return x[:, -1, :]

    
    
class UnifiedPhasedLSTMModel(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 class_dim, 
                 output_dim, 
                 nlayers,
                 linear_period=12*5,
                 comp_period=12):
        super(UnifiedPhasedLSTMModel, self).__init__()
        
        self.class_dim = class_dim
        self.training_mode = -1
        self.class_value = torch.Tensor([-1]).to(device)
        self.test_mode = 0
        self.tic = 0
        self.classifier_weight = 0
        self.linear_period = linear_period
        self.comp_period = comp_period
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)
        
        
        self.classifier = nn.Linear(input_dim, output_dim)
        
        self.comp_in = PhasedLSTM(input_dim, hidden_dim)
        self.comp_out = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        
        def simple(x):
            x = self.fc_in(x)
            x = self.relu(x)
            x = self.fc_out(x)
            return x
        
        def component(x):
            x, _ = self.comp_in(x)
            x = self.comp_out(x)
            return x
        
        def classifier(x):
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x
        
        # single model training
        if self.training_mode == 0:
            x = component(x) if self.class_value > 0.5 else simple(x)
        
        # classifer training
        elif self.training_mode == 1:
            x = classifier(x)

        # unified model training
        else:
            if isinstance(self.classifier_weight, int):
                self.classifier_weight = classifier(x)
            
            if self.test_mode == 1:
                if self.classifier_weight.mean() > 0.5:
                    if self.tic % self.comp_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = component(x)
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = simple(x)
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
                x = torch.where(self.classifier_weight.mean() > 0.5, component(x), simple(x))
        
        return x[:, -1, :]
    

class UnifiedTransformerModel(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 class_dim, 
                 output_dim, 
                 nlayers,
                 linear_period=12*5,
                 comp_period=12):
        super(UnifiedTransformerModel, self).__init__()
        
        self.class_dim = class_dim
        self.training_mode = -1
        self.class_value = torch.Tensor([-1]).to(device)
        self.test_mode = 0
        self.tic = 0
        self.classifier_weight = 0
        self.linear_period = linear_period
        self.comp_period = comp_period
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)
        
        
        self.classifier = nn.Linear(input_dim, output_dim)
        
        self.comp_in = TransformerEncoder(output_size=1, d_model=hidden_dim, nhead=4, hidden_size=hidden_dim, num_layers=1).to(device)
        self.comp_out = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        
        def simple(x):
            x = self.fc_in(x)
            x = self.relu(x)
            x = self.fc_out(x)
            return x
        
        def component(x):
            x = self.comp_in(x)
            x = self.comp_out(x)
            return x
        
        def classifier(x):
            x = self.classifier(x)
            x = self.sigmoid(x)
            return x
        
        # single model training
        if self.training_mode == 0:
            x = component(x) if self.class_value > 0.5 else simple(x)
        
        # classifer training
        elif self.training_mode == 1:
            x = classifier(x)

        # unified model training
        else:
            if isinstance(self.classifier_weight, int):
                self.classifier_weight = classifier(x)
            
            if self.test_mode == 1:
                if self.classifier_weight.mean() > 0.5:
                    if self.tic % self.comp_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = component(x)
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                    x = simple(x)
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
                x = torch.where(self.classifier_weight.mean() > 0.5, component(x), simple(x))
        
        return x[:, -1, :]
    


def check_inference_time(model, mode, testX, testy, repetitions=10):
    
    model.training_mode = mode
    model.test_mode = 1
    if mode == 0:
        model.class_value = torch.Tensor([1]).to(device)
    if mode == 1:
        model.class_value = torch.Tensor([-1]).to(device)
    model.tic = 0
    
    model.eval()
    model_timings=np.zeros((repetitions,1))
    
    for _ in range(10):
        _ = model(torch.unsqueeze(testX[0], 0))
    
    with torch.no_grad(): 
        for rep in range(repetitions):

            model.tic = 0
            starter = time()
            for pr in range(len(testX)):
                predicted = model(torch.unsqueeze(testX[pr], 0))
            ender = time()
                            
            curr_time = (ender - starter) / (len(testX))
            
            model_timings[rep] = curr_time
            print('#' + str(rep + 1) + ': ', curr_time, 'ms')
            
            
    print('Inference time : ', np.average(np.array(model_timings)), 'ms')
    return np.average(np.array(model_timings))


