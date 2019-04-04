# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:40:34 2019

@author: Zymieth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# predictor class 
class MLP(nn.Module):
    def __init__(self, dense_layers, softmax = True):
        '''
        Simple dense MLP class used as predictor for HDNN
        
        :smax_l: leave True for softmax applied to ouput
        '''
        super().__init__()
        self.dense_layers = nn.ModuleList([nn.Linear(dense_layers[i], dense_layers[i + 1]) \
                                           for i in range(len(dense_layers) - 1)])
        self.softmax = softmax
        
    def getLength(self):
        return len(self.dense_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.dense_layers:
            l_x = l(x)
            x = F.softplus(l_x, beta=10,threshold=20)#F.relu(l_x)#torch.sigmoid(l_x)#l_x# 
        if not self.softmax: return l_x
        else: return F.log_softmax(l_x, dim=-1)

    
class CNN(nn.Module):
    def __init__(self, conv_layers, dense_layers, smax_l = True):
        '''
        smax_l: leave True for softmax applied to ouput
        '''
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv1d(conv_layers[i], conv_layers[i + 1], kernel_size = 3) 
                                     for i in range(len(conv_layers) - 1)])
        self.dense_layers = nn.ModuleList([nn.Linear(dense_layers[i], dense_layers[i + 1]) 
                                     for i in range(len(dense_layers) - 1)])

        self.max = nn.MaxPool1d(2)
        self.smax = smax_l

        
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        #if self.training():
            #x = self.noise(x)
        for i,l in enumerate(self.conv_layers):
            x = l(x)
            #x = self.bn[i](x)
            x = F.relu(x)
            #x = self.max(x)
        x = x.view(x.size(0), -1)
        for l in self.dense_layers:
            l_x = l(x)
            x = F.relu(l_x)
        if self.smax: return F.log_softmax(l_x, dim=-1)
        else: return torch.sigmoid(l_x)