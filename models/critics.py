import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (STATE_SIZE, SEED,
                      ACTION_SIZE, LAYER_C1,
                      LAYER_C2)
import numpy as np
torch._dynamo.config.suppress_errors = False

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    

class Critic1(nn.Module):
    def __init__(self, 
                 input=STATE_SIZE+ACTION_SIZE,
                 seed=SEED,
                 layers=LAYER_C1):
        super(Critic1, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(input, layers[0])
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        self.layer_2 = nn.Linear(layers[0], layers[2])
        self.batch_norm_2 = nn.BatchNorm1d(layers[2])
        self.layer_3 = nn.Linear(layers[2], 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_2.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state_action):
        x = F.relu(self.layer_1(state_action))
        x = F.relu(self.batch_norm_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.batch_norm_2(x))
        return self.layer_3(x)

    

class Critic2(nn.Module):
    def __init__(self, 
                 input=STATE_SIZE + ACTION_SIZE,
                 seed=SEED,
                 layers=LAYER_C1):
        super(Critic2, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(input, layers[0])
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        self.layer_2 = nn.Linear(layers[0], layers[2])
        self.batch_norm_2 = nn.BatchNorm1d(layers[2])
        self.layer_3 = nn.Linear(layers[2], 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_2.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state_action):
        x = F.relu(self.layer_1(state_action))
        x = F.relu(self.batch_norm_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.batch_norm_2(x))
        return self.layer_3(x)
