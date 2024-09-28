import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (S_SIZE, SEED, A_SIZE)
import numpy as np
torch._dynamo.config.suppress_errors = False

class Critic(nn.Module):
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, seed=SEED,layers=[100, 50]):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer_s0 = nn.Linear(state_size, layers[0])
        self.batch_norm_s0 = nn.LayerNorm(layers[0])
        
        self.layer_s1 = nn.Linear(layers[0], layers[1])
        self.batch_norm_s1 = nn.LayerNorm(layers[1])
        
        self.layer_a  = nn.Linear(action_size, layers[1])
        self.batch_norm_a = nn.LayerNorm(layers[1])
        
        self.layer_sa = nn.Linear(layers[1], layers[2])
        self.batch_norm_sa = nn.LayerNorm(layers[2])
        
        self.layer_out=nn.Linear(layers[2],1)
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in [self.layer_s0, self.layer_s1, self.layer_a, self.layer_sa ]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
        
    def forward(self, state, action ):
        s0 = F.relu(self.batch_norm_s0(self.layer_s0(state)))
        s = F.relu(self.batch_norm_s1(self.layer_s1(s0)))
        a = F.relu(self.batch_norm_a(self.layer_a(action)))
        sa = F.relu(self.batch_norm_sa(self.layer_sa(torch.add(s,a))))
        q_val = self.layer_out(F.leaky_relu(sa))
        return q_val     


