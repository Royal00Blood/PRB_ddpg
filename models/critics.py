import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (STATE_SIZE, SEED,
                      ACTION_SIZE, LAYER_C1,
                      LAYER_C2)
import numpy as np
torch._dynamo.config.suppress_errors = True

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
class Critic1(nn.Module):
    def __init__(self, state_size= STATE_SIZE,
                 action_size=ACTION_SIZE,
                 seed= SEED,
                 layers =LAYER_C1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fs1 = nn.Linear(state_size, layers[0])
        self.fa1 = nn.Linear(action_size, layers[0])
        self.fc2 = nn.Linear(layers[0]*2, layers[1])
        self.fc3 = nn.Linear(layers[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3.weight.data.normal_(0,0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Critic2(nn.Module):
    def __init__(self, state_size= STATE_SIZE, 
                 action_size=ACTION_SIZE, seed= SEED, layers=LAYER_C2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, layers[0])
        self.fc2 = nn.Linear(layers[0]+action_size, layers[1])
        self.fc3 = nn.Linear(layers[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)