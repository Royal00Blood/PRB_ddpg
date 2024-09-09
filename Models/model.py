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
    
class Actor(nn.Module):
    def __init__(self, 
                 state_size = STATE_SIZE, 
                 action_size=ACTION_SIZE, 
                 seed=SEED,
                 layers=LAYER_A ):
        super(Actor, self).__init__()
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
    
class Critic1(nn.Module):
    def __init__(self, state_size= STATE_SIZE, action_size=ACTION_SIZE, seed= SEED, fcs1_units=C_FC11, fc2_units=C_FC12):
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
        self.fs1 = nn.Linear(state_size, fcs1_units)
        self.fa1 = nn.Linear(action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units*2, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
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
                 action_size=ACTION_SIZE, seed= SEED, fcs1_units=C_FC21, fc2_units=C_FC22):
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
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
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