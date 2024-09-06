import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import ACTION_, STATE_SIZE, SEED, ACTION_SIZE,LAYER_A,LAYER_C1,LAYER_C2
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
        
        self.fc3v = nn.Linear(layers[1], layers[2])
        self.fc3w = nn.Linear(layers[1], layers[2])
        
        self.fc5v = nn.Linear(layers[2], layers[3])
        self.fc5w = nn.Linear(layers[2], layers[3])
        self.bn5 = nn.LayerNorm(layers[3])
        
        self.fc6v = nn.Linear(layers[3], 1)
        self.fc6w = nn.Linear(layers[3], 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        self.fc3v.weight.data.uniform_(*hidden_init(self.fc3v))
        self.fc5v.weight.data.uniform_(*hidden_init(self.fc5v))
        self.fc6v.weight.data.uniform_(-0.01, 0.01)
        
        self.fc3w.weight.data.uniform_(*hidden_init(self.fc3w))
        self.fc5w.weight.data.uniform_(*hidden_init(self.fc5w))
        self.fc6w.weight.data.uniform_(-0.01, 0.01)
        
    def forward(self, state):
    
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        v = F.relu(self.fc3v(x))
        v = F.relu(self.fc5v(v))
        v = self.bn5(v)
        v = F.tanh(self.fc6v(v))
    
        w = F.relu(self.fc3w(x))
        w = F.relu(self.fc5w(w))
        w = self.bn5(w)
        w = F.tanh(self.fc6w(w))
 
        return torch.cat([v, w], dim=-1)*ACTION_
    
class Critic1(nn.Module):
    def __init__(self, 
                 state_size=STATE_SIZE,
                 action_size=ACTION_SIZE, 
                 seed= SEED, 
                 layer=LAYER_C1):
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
        self.fs1 = nn.Linear(state_size, layer[0])
        self.fa1 = nn.Linear(action_size, layer[0])
        self.fc2 = nn.Linear(layer[0]+layer[0], layer[1])
        self.fc3 = nn.Linear(layer[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fs1.weight.data.normal_(0,0.1)
        self.fa1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        s = F.relu(self.fs1(state))
        a = F.relu(self.fa1(action))
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
    
class Critic2(nn.Module):
    def __init__(self, 
                 state_size=STATE_SIZE,
                 action_size=ACTION_SIZE, 
                 seed= SEED, 
                 layer=LAYER_C2):   
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
        self.fs1 = nn.Linear(state_size, layer[0])
        self.fa1 = nn.Linear(action_size, layer[0])
        self.fc2 = nn.Linear(layer[0]*2, layer[1])
        self.fc3 = nn.Linear(layer[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fs1.weight.data.normal_(0,0.1)
        self.fa1.weight.data.normal_(0,0.1)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        s = F.relu(self.fs1(state))
        a = F.relu(self.fa1(action))
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))