import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (STATE_SIZE, SEED,
                      ACTION_SIZE, LAYER_C1,
                      LAYER_C2)
import numpy as np
torch._dynamo.config.suppress_errors = False

class Critic(nn.Module):
    def __init__(self, 
                 state_size=STATE_SIZE,
                 action_size=ACTION_SIZE,
                 seed=SEED,
                 layers=LAYER_C1):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_s0 = nn.Linear(state_size, layers[0])
        self.layer_s1 = nn.Linear(layers[0], layers[1])
        self.layer_a  = nn.Linear(action_size, layers[1])
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.layer_s0.weight, 0., 0.1)
        nn.init.constant_(self.layer_s0.bias, 0.1)
        nn.init.normal_(self.layer_s1.weight, 0., 0.1)
        nn.init.constant_(self.layer_s1.bias, 0.1)
        nn.init.normal_(self.layer_a.weight, 0., 0.1)
        nn.init.constant_(self.layer_a.bias, 0.1)
        
    def forward(self, state, action ):
        s = F.relu(self.layer_s0(state))
        s = F.relu(self.layer_s1(s)) 
        a = F.relu(self.layer_a(action))
        q_val = self.output(torch.relu(s+a))
        return q_val     

class Critic3(nn.Module):
    def __init__(self, 
                 input=STATE_SIZE + ACTION_SIZE,
                 seed=SEED,
                 layers=LAYER_C1):
        super(Critic3, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(input, layers[0])
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        self.layer_2 = nn.Linear(layers[0], layers[1])
        self.batch_norm_2 = nn.BatchNorm1d(layers[1])
        self.layer_3 = nn.Linear(layers[1], 1)
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
    
class Critic4(nn.Module):
    def __init__(self, 
                    input=STATE_SIZE + ACTION_SIZE,
                    seed=SEED,
                    layers=LAYER_C2):
        super(Critic4, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(input, layers[0])
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        self.layer_2 = nn.Linear(layers[0], layers[1])
        self.batch_norm_2 = nn.BatchNorm1d(layers[1])
        self.layer_3 = nn.Linear(layers[1], 1)
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
