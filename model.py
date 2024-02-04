import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, inputs):
        energy = torch.tanh(inputs)
        energy = torch.matmul(energy, self.attn_weights).squeeze(dim=-1)
        attention_weights = torch.softmax(energy, dim=-1).unsqueeze(dim=-1)
        context_vector = torch.sum(inputs * attention_weights, dim=1)
        return context_vector
    
class ConvNet_BiLSTM(torch.nn.Module):
    def __init__(self, output_dim,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size,wordvec_len):
        super(ConvNet_BiLSTM, self).__init__()
        self.conv = torch.nn.Sequential()
        
        # cnn layers
        for i in range(len(filter_size.split('-'))):
            if i == 0:
                self.conv.add_module("conv_" + str(i + 1),
                                     torch.nn.Conv1d(wordvec_len, int(filter_num.split('-')[i]),
                                                     kernel_size=int(filter_size.split('-')[i]),
                                                     padding=int(int(filter_size.split('-')[i])/2)))
            else:
                self.conv.add_module("conv_" + str(i + 1),
                                     torch.nn.Conv1d(int(filter_num.split('-')[i - 1]),
                                                     int(filter_num.split('-')[i]),
                                                     kernel_size=int(filter_size.split('-')[i])))
            # pool
            if pool_size != 0:
                self.conv.add_module("maxpool_" + str(i + 1), torch.nn.MaxPool1d(kernel_size=pool_size))

            # activation
            self.conv.add_module("relu_" + str(i + 1), torch.nn.ReLU())

            # batchnorm
            if if_bn == 'Y':
                self.conv.add_module("batchnorm_" + str(i + 1),
                                     torch.nn.BatchNorm1d(int(filter_num.split('-')[i])))

            # dropout
            self.conv.add_module("dropout_" + str(i + 1), torch.nn.Dropout(cnndrop_out))
            
        self.lstm = torch.nn.LSTM(int(filter_num.split('-')[-1]), rnn_size, 1, batch_first=True, bidirectional=True)
        
        # Attention layer
        self.attention = AttentionLayer(rnn_size * 2)
        
        #fc layer
        self.fc = torch.nn.Sequential()

        if fc_size >0:
            self.fc.add_module("fc_1", torch.nn.Linear(rnn_size * 2, int(fc_size)))
            self.fc.add_module("relu_1", torch.nn.ReLU())
            self.fc.add_module("fc_2", torch.nn.Linear(int(fc_size), output_dim))
        else:
            self.fc.add_module("fc_1",torch.nn.Linear(rnn_size * 2, output_dim))

    def forward(self, x,filter_size,filter_num,pool_size,if_bn,cnndrop_out,rnn_size,fc_size):
        h0 = Variable(torch.zeros(1 * 2, x.size(0), rnn_size)).to(device) 
        c0 = Variable(torch.zeros(1 * 2, x.size(0), rnn_size)).to(device)
        x = x.transpose(1, 2)
        x = self.conv.forward(x)
        x= x.transpose(1,2)
        out, _ = self.lstm(x, (h0, c0))
        attended_out = self.attention(out)
        out = self.fc(attended_out)
        #out = self.fc(torch.mean(out, 1))
        return out




