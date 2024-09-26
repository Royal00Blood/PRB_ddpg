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
        self.output = nn.Linear(layers[1], 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.layer_s0.weight, 0., 0.1)
        nn.init.constant_(self.layer_s0.bias, 0.1)
        nn.init.normal_(self.layer_s1.weight, 0., 0.1)
        nn.init.constant_(self.layer_s1.bias, 0.1)
        nn.init.normal_(self.layer_a.weight, 0., 0.1)
        nn.init.constant_(self.layer_a.bias, 0.1)
        
    def forward(self, state, action ):
        s0 = F.relu(self.batch_norm_s0(self.layer_s0(state)))
        s = F.relu(self.batch_norm_s1(self.layer_s1(s0)))
        a = F.relu(self.batch_norm_a(self.layer_a(action)))
        q_val = self.output(torch.relu(s+a))
        return q_val     


