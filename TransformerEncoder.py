import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args: x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #print(x.shape, self.pe[:x.shape[0], :].shape)
        x += self.pe[:x.shape[0], :]
        return self.dropout(x)
        

class TransformerEncoder(nn.Module):
    def __init__(self, output_size, d_model, nhead, hidden_size, num_layers):
        super(TransformerEncoder, self).__init__()
        
        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #self.fc = nn.Linear(d_model, output_size, bias=True)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.te(x)#.transpose(0, 1)
        #x = self.fc(x)[-1]
        #print(x.shape)
        return x

