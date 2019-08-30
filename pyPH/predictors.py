import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dense_layers, softmax = True):
        """Simple dense MLP class used as predictor"""
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
            x = F.softplus(l_x, beta=10,threshold=20)
        if not self.softmax: return l_x
        else: return F.log_softmax(l_x, dim=-1)


