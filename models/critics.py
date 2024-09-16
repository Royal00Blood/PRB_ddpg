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
    def __init__(self, state_size=STATE_SIZE,
                 action_size=ACTION_SIZE,
                 seed=SEED,
                 layers=LAYER_C1):
        super(Critic1, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_s = nn.Linear(STATE_SIZE, layers[0])
        self.layer_a = nn.Linear(ACTION_SIZE, layers[0])
        
        #print(f"self.layer_s {self.layer_s}, self.layer_a {self.layer_a}")
        self.layer_as = nn.Linear(layers[0] * 2, layers[0])
        #print(f"self.layer_s {self.layer_s}, self.layer_a {self.layer_a} , self.layer_as {self.layer_as}")
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        
        self.layer2 = nn.Linear(layers[0], layers[1])
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        
        self.layer3 = nn.Linear(layers[1], 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.layer_s.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_a.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_as.weight.data.uniform_(-3e-3, 3e-3)
        self.layer2.weight.data.uniform_(-3e-3, 3e-3)
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        state_features = F.relu(self.layer_s(state))
        action_features = F.relu(self.layer_a(action))
        # print(f"state_features.size {state_features.size()}, action_features {action_features.size()}")
        x = F.relu(self.layer_as(torch.cat((state_features, action_features), 1)))
        x = F.relu(self.batch_norm2(self.layer2(x)))
        return self.layer3(x)

    

class Critic2(nn.Module):
    def __init__(self, state_size=STATE_SIZE,
                 action_size=ACTION_SIZE,
                 seed=SEED,
                 layers=LAYER_C2):
        super(Critic2, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Linear(state_size + action_size, layers[0])
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        
        self.layer2 = nn.Linear(layers[0], layers[1])
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        
        self.layer3 = nn.Linear(layers[1], 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')
        
        
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = F.relu(self.batch_norm2(self.layer2(x)))
        return self.layer3(x)
