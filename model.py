import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import ACTION_, STATE_SIZE, SEED
import numpy as np
torch._dynamo.config.suppress_errors = True

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 300)
        
        self.fc3v = nn.Linear(300, 350)
        self.fc3w = nn.Linear(300, 350)
        
        self.fc4v = nn.Linear(350, 400)
        self.fc4w = nn.Linear(350, 400)
        
        self.fc5v = nn.Linear(400, 512)
        self.fc5w = nn.Linear(400, 512)
        self.bn5 = nn.LayerNorm(512)
        
        self.fc6v = nn.Linear(512, 1)
        self.fc6w = nn.Linear(512, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        self.fc3v.weight.data.uniform_(*hidden_init(self.fc3v))
        self.fc4v.weight.data.uniform_(*hidden_init(self.fc4v))
        self.fc5v.weight.data.uniform_(*hidden_init(self.fc5v))
        self.fc6v.weight.data.uniform_(-3e-3, 3e-3)
        
        self.fc3w.weight.data.uniform_(*hidden_init(self.fc3w))
        self.fc4w.weight.data.uniform_(*hidden_init(self.fc4w))
        self.fc5w.weight.data.uniform_(*hidden_init(self.fc5w))
        self.fc6w.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
    
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        v = F.relu(self.fc3v(x))
        v = F.relu(self.fc4v(v))
        v = F.relu(self.fc5v(v))
        v = self.bn5(v)
        v = F.tanh(self.fc6v(v))
        
        w = F.relu(self.fc3w(x))
        w = F.relu(self.fc4w(w))
        w = F.relu(self.fc5w(w))
        w = self.bn5(w)
        w = F.tanh(self.fc6w(w))
        
        return torch.cat([v*ACTION_, w*ACTION_], dim=-1)
    
class Critic1(nn.Module):
    def __init__(self, state_size= STATE_SIZE, action_size=ACTION_, seed= SEED, fcs1_units=400, fc2_units=300):
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
    
class Critic2(nn.Module):
    def __init__(self, state_size= STATE_SIZE, action_size=ACTION_, seed= SEED, fcs1_units=450, fc2_units=350):
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