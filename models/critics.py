import torch
import os
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (S_SIZE, SEED, A_SIZE, DIR_CHEKPOINT,NUM_PARAMETERS,INIT)
import numpy as np
torch._dynamo.config.suppress_errors = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, seed=SEED,layers=[100, 50], dir_chekpoint= DIR_CHEKPOINT, name = "critic_checpoint"):
        super(Critic, self).__init__()
        
        self.chekpoint = os.path.join(dir_chekpoint, name)
        self.seed = torch.manual_seed(seed)
        self.relu =nn.LeakyReLU() #nn.PReLU(num_parameters=NUM_PARAMETERS, init=INIT, device=device)#nn.ReLU()
        self.layer_1 = nn.Linear(state_size+action_size, layers[0])
        self.batch_norm_1 = nn.LayerNorm(layers[0])
        self.layer_2 = nn.Linear(layers[0], layers[1])
        self.batch_norm_2 = nn.LayerNorm(layers[1])
        self.layer_3 = nn.Linear(layers[1], layers[2])
        self.batch_norm_3 = nn.LayerNorm(layers[2])
        self.layer_4 = nn.Linear(layers[2], layers[3])
        self.batch_norm_4 = nn.LayerNorm(layers[3])
        self.q = nn.Linear(layers[3], 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in [self.layer_1, self.layer_2,  self.layer_3, self.layer_4]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    
            # Инициализируем смещения
            nn.init.constant_(layer.bias, 0.1)

            # Проверяем, является ли слой PReLU и если да, то инициализируем альфа
            if isinstance(layer, nn.LeakyReLU):#nn.PReLU):
                nn.init.constant_(layer.weight, INIT)
                
    def forward(self, state_action ):
        layer_1 = self.relu(self.layer_1(state_action))
        layer_2 = self.relu(self.layer_2(layer_1))
        layer_3 = self.relu(self.layer_3(layer_2))
        layer_4 = self.relu(self.layer_4(layer_3))
        q_val = self.q(layer_4)
        return q_val 
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chekpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chekpoint))    


