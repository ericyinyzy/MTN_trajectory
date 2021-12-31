# -*- coding: utf-8 -*-
# date: 2018-11-30 17:00
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalDecoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalDecoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe=pe.repeat(128,1,1)
        self.register_buffer('pe', pe)
        self.query_embed = nn.Embedding(45, d_model)
        self.lut = nn.Linear(65, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print('xxx',x.shape,self.pe.shape,self.pe[:, :x.size(1)].shape)
        # if encode:
        # print(x.shape)
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        # print('input',x.shape,self.pe[:,:x.size(1)].shape)
        posEmbed=self.pe[:, :x.size(1)].repeat(x.size(0),1,1)
        x=torch.cat((Variable(posEmbed, requires_grad=False),x),axis=-1)
        # x=self.lut(x)
        # print(x.shape)
        # print('dec_inputshape',x.shape)
        # exit()

        # else:
        #     query_embed = self.query_embed.unsqueeze(0)
        #     x=x+Variable(query_embed, requires_grad=True)
        # print('shapeeee',self.pe[:, :x.size(1)].shape,x.shape)
        # exit()
        # else:
        #     query_embed = self.query_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

        return self.dropout(x)
class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) /d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.query_embed = nn.Embedding(45, d_model)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)

