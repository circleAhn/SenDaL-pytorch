##
## Models.py
## 시계열 처리를 위한 모델
##

import torch
from torch import nn
from PhasedLSTM import PhasedLSTM
from TransformerEncoder import TransformerEncoder
import numpy as np
from time import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnifiedLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_dim, output_dim, nlayers):
        super(UnifiedLSTM, self).__init__()
        
        self.class_dim = class_dim
        self.training_mode = -1
        self.class_value = torch.Tensor([-1]).to(device)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)
        self.classifier = nn.Linear(input_dim, output_dim)
        
        #self.lstm_ebd = nn.Linear(input_dim, hidden_dim, bias=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=nlayers, batch_first=True)
        self.lstm_fc = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.fc_ebd = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        
        # single model training
        if self.training_mode == 0:
            if self.class_value > 0:
                #x = self.lstm_ebd(x)
                x, _ = self.lstm(x)
                x = self.lstm_fc(x)
            else:
                x = self.fc_ebd(x)
                x = self.relu(x)
                x = self.fc(x)
        
        # classifer training
        elif self.training_mode == 1:
            cls = self.classifier(x)
            x = self.sigmoid(cls)
        
        # unified model training
        else:
            cls = self.classifier(x)
            cls = 2 * self.sigmoid(cls) - 1
            if cls.mean() > 0:
                #x = self.lstm_ebd(x)
                x, _ = self.lstm(x)
                x = self.lstm_fc(x)
            else:
                x = self.fc_ebd(x)
                x = self.relu(x)
                x = self.fc(x)
        
        return x[:, -1, :]

    
    
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
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
            
            torch.cuda.synchronize()
            x = component(x) if self.classifier_weight.mean() > 0.5 else simple(x)
        
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
        self.class_value = torch.Tensor([1]).to(device)
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
            #x = component(x) if self.class_value > 0.5 else simple(x)
        
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
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
            
            x = component(x) if self.classifier_weight.mean() > 0.5 else simple(x)
            #x = torch.where(self.classifier_weight.mean() > 0.5, component(x), simple(x))
            #x = component(x) if self.classifier_weight.mean() > 0.5 else simple(x)
        
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
        self.class_value = torch.Tensor([1]).to(device)
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
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
            
            x = component(x) if self.classifier_weight.mean() > 0.5 else simple(x)
        
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
        self.class_value = torch.Tensor([1]).to(device)
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
                else:
                    if self.tic % self.linear_period == 0:
                        self.classifier_weight = classifier(x)
                        self.tic = 0
                self.tic += 1
            else:
                self.classifier_weight = classifier(x)
            
            x = component(x) if self.classifier_weight.mean() > 0.5 else simple(x)
        
        return x[:, -1, :]
    
    
def model_training(model, 
                   train_df,
                   valid_df,
                   mode=0, #single=0, classify=1, unif=-1
                   num_epochs=1000, 
                   lr=0.02, 
                   verbose=50, 
                   patience=20):
    
    def training(num_epochs=num_epochs):
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_hist = np.zeros(num_epochs)
        valid_hist = np.zeros(num_epochs)
        trigger_times = 0
        
        for epoch in range(num_epochs):
            model.train()
            avg_loss = 0

            for batch_idx, samples in enumerate(train_df):
                X_train, y_train = samples
                X_train = X_train.to(device)
                y_train = y_train.to(device)

                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                avg_loss += loss / len(train_df)

            train_hist[epoch] = avg_loss
            valid_hist[epoch] = model_valid(model, mode, valid_df)

            if epoch % verbose == 0:
                print('Epoch:', '%04d'%(epoch), 'train loss:', '{:.4f}'.format(avg_loss), 'valid loss:', '{:.4f}'.format(valid_hist[epoch]))
                
            if epoch > 200:
                if valid_hist[epoch] + 1e-2 >= valid_hist[epoch - 1]:
                    trigger_times += 1
                    
                    if trigger_times >= patience:
                        print('\n Early Stopping')
                        break
                else:
                    trigger_times = 0
                    
                    
        return model.eval(), train_hist
    
    model.training_mode = mode
    model.test_mode = 0
    
    if mode == 0: #single
        freeze_parameters(model, ['classifier.weight', 'classifier.bias'])
        criterion = nn.MSELoss().to(device)
        
        model.class_value = -1
        model, train_hist = training()
        model.class_value = 1
        model, train_hist = training()
        
        return model, train_hist
        
    if mode == 1: #classify
        param_list = []
        for name, _ in model.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param_list.append(name)
        freeze_parameters(model, param_list)

        criterion = nn.BCELoss().to(device)
        model, train_hist = training()
        return model, train_hist
    
    if mode != -1:
        return None, None
    
    #unified
    param_list = []
    for name, _ in model.named_parameters():
        if name not in ['comp_out.weight', 'comp_out.bias', 'fc_out.weight', 'fc_out.bias']:
            param_list.append(name)
    freeze_parameters(model, param_list)
    criterion = nn.MSELoss().to(device)
    model, train_hist = training()
    
    return model.eval(), train_hist


def MAE(true, pred):
       return np.mean(np.abs(np.array(true).ravel()-np.array(pred).ravel()))
    
    
def model_valid(model, mode, valid_df):

    model.eval()
    criterion = nn.BCELoss().to(device) if mode == 1 else nn.MSELoss().to(device)
    avg_loss = 0
    with torch.no_grad():
        for batch_idx, samples in enumerate(valid_df):
            X_valid, y_valid = samples
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            outputs = model(X_valid)
            loss = criterion(outputs, y_valid)
        avg_loss += loss / len(valid_df)
    
    return avg_loss




def model_predict(model, testX, ground_truth, 
                       mode=0, class_value=-1):
    
    model.eval()
    model.training_mode = mode
    model.test_mode = 1
    if mode == 0:
        model.class_value = class_value
    model.tic = 0
    
    pred = np.zeros(len(testX))

    with torch.no_grad():
        for i in range(len(testX)):    
            predicted = model(torch.unsqueeze(torch.Tensor(testX[i]).to(device), 0))
            predicted = torch.flatten(predicted).item()
            pred[i] = predicted
    
    return MAE(ground_truth, pred), pred


def freeze_parameters(model, freeze_list=None):
    
    assert isinstance([], list), 'Parameter "freeze_list" must be type of list'
    for param in model.parameters():
        param.requires_grad = True
    
    if freeze_list is not None:
        for name, param in model.named_parameters():
            if name in freeze_list:
                param.requires_grad = False
                
                
                
def check_inference_time(model, mode, testX, testy, device, repetitions=10):
    
    def MAE(true, pred):
        return np.mean(np.abs(np.array(true).ravel()-np.array(pred).ravel()))
    
    model.training_mode = mode
    model.test_mode = 1
    if mode == 0:
        model.class_value = torch.Tensor([1]).to(device)
    if mode == 1:
        model.class_value = torch.Tensor([-1]).to(device)
    model.tic = 0
    
    
    model.eval()
    model_timings=np.zeros((repetitions,1))
    for _ in range(100):
        _ = model(torch.unsqueeze(testX[0], 0))
    
    with torch.no_grad(): 
        for rep in range(repetitions):

            starter = time()
            for pr in range(len(testX)):
                predicted = model(torch.unsqueeze(testX[pr], 0))                  
            ender = time()

            curr_time = (ender - starter) / (len(testX))
            
            
            model_timings[rep] = curr_time
            print('#' + str(rep + 1) + ': ', curr_time, 'ms')

    print('Inference time : ', np.average(np.array(model_timings)), 'ms')
    
    return np.average(np.array(model_timings))


