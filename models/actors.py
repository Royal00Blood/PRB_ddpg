import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (ACTION_, STATE_SIZE, SEED,
                      ACTION_SIZE,LAYER_A,LAYER_C1
                      ,LAYER_C2)
import numpy as np
torch._dynamo.config.suppress_errors = True

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
class Actor_1(nn.Module):
    def __init__(self, 
                 state_size = STATE_SIZE, 
                 action_size=ACTION_SIZE, 
                 seed=SEED,
                 layers=LAYER_A ):
        super(Actor_1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3.weight.data.normal_(0,0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.tanh(self.fc4(x))*ACTION_
        return action

class Actor_2(nn.Module):
    def __init__(self, 
                 state_size = STATE_SIZE, 
                 action_size=ACTION_SIZE, 
                 seed=SEED,
                 layers=LAYER_A ):
        super(Actor_2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3.weight.data.normal_(0,0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.tanh(self.fc4(x))*ACTION_
        return action
    
