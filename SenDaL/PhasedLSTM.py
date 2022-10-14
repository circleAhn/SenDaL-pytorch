"""
phased LSTM
https://arxiv.org/pdf/1610.09513v1.pdf
"""

import torch
from torch import nn


class PhasedLSTMCell(nn.Module):
    
    def __init__(
        self,
        hidden_size,
        alpha=0.001
    ):
        
        super(PhasedLSTMCell, self).__init__()

        self.hidden_size = hidden_size
        self.alpha = alpha
        
        period = torch.Tensor(hidden_size).uniform_(10.0, 100.0)
        shift = torch.Tensor(hidden_size).uniform_(0.0, 1000.0)
        r_on = torch.zeros(hidden_size) + 0.5
        
        self.period = nn.Parameter(period)
        self.shift = nn.Parameter(shift)
        self.r_on = nn.Parameter(r_on)



    def forward(self, h_hat, c_hat, t):
        
        period = self.period
        shift = self.shift
        r_on = self.r_on
        
        phi = ((t - shift) - (period * ((t - shift) // period))) / period
        
        k = torch.where(r_on < phi, self.alpha * phi, 2 * phi / r_on)
        k = torch.where(phi < 0.5 * r_on, k, 2 - 2 * phi / r_on)
        
        c = k * c_hat + (1 - k) * self.c_0
        h = k * h_hat + (1 - k) * self.h_0

        return h, c


class PhasedLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super(PhasedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.t = 0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.plstm_cell = PhasedLSTMCell(
            hidden_size=hidden_size
        )

        
    def forward(self, x):
        
        t = self.t
        self.t = t + 1
        
        h_prev = x.new_zeros((1, x.size(0), self.hidden_size))
        c_prev = x.new_zeros((1, x.size(0), self.hidden_size))
        
        self.plstm_cell.h_0 = h_prev
        self.plstm_cell.c_0 = c_prev

        outputs = []
        for i in range(x.size(1)):
            u_t = x[:, i, :].unsqueeze(1)

            out, (h_t, c_t) = self.lstm(u_t, (h_prev, c_prev))
            #h_0, c_0 = self.plstm_cell(h_t, c_t, t+i)
            
            self.plstm_cell.h_0 = h_t
            self.plstm_cell.c_0 = c_t
            
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)

        return outputs, (self.plstm_cell.h_0, self.plstm_cell.c_0)