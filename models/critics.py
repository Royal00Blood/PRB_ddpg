import torch
import os
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (S_SIZE, SEED, A_SIZE, DIR_CHEKPOINT)
import numpy as np
torch._dynamo.config.suppress_errors = False

class Critic(nn.Module):
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, seed=SEED,layers=[100, 50], dir_chekpoint= DIR_CHEKPOINT, name = "critic_checpoint"):
        super(Critic, self).__init__()
        
        self.chekpoint = os.path.join(dir_chekpoint, name)
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(state_size+action_size, layers[0])
        self.batch_norm_1 = nn.LayerNorm(layers[0])
        self.layer_2 = nn.Linear(layers[0], layers[1])
        self.batch_norm_2 = nn.LayerNorm(layers[1])
        self.layer_3 = nn.Linear(layers[1], layers[2])
        self.batch_norm_3 = nn.LayerNorm(layers[2])
        self.q = nn.Linear(layers[2], 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in [self.layer_1, self.layer_2]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
        
    def forward(self, state_action ):
        layer_1 = F.relu(self.batch_norm_1(self.layer_1(state_action)))
        layer_2 = F.relu(self.batch_norm_2(self.layer_2(layer_1)))
        layer_3 = F.relu(self.batch_norm_3(self.layer_3(layer_2)))
        q_val = self.q(layer_3)
        return q_val 
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chekpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chekpointe))    


