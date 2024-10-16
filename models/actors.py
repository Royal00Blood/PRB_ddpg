import torch
import os
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (A_MAX, S_SIZE, A_SIZE, SEED, L_A, DIR_CHEKPOINT)
torch._dynamo.config.suppress_errors = False
    
class Actor(nn.Module):
    """ Арихитектура модели Актера для сети DDPG принимает на вход:
        state_size  - размерность вектора состояния 
        action_size - размерность вектора действия
        seed        - параметр "скорости" обучения
        action_max  - максимальное значение действия по модулю
        layers      - вектор со значениями количества нейронов в каждом слое
    """
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, seed=SEED, action_max = A_MAX, 
                 layers=L_A, dir_chekpoint=DIR_CHEKPOINT, name = 'chekpoint_actor' ):
        super(Actor, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.chekpoint = os.path.join(dir_chekpoint, name)
        self.seed = torch.manual_seed(seed)
        self.action_max = action_max
        
        # init layers
        self.layer_1 = nn.Linear(state_size, layers[0])
        self.batch_norm_1 = nn.LayerNorm(layers[0])
        
        self.layer_2 = nn.Linear(layers[0], layers[1])
        self.batch_norm_2 = nn.LayerNorm(layers[1])
        
        self.layer_3 = nn.Linear(layers[1], layers[2])
        self.batch_norm_3 = nn.LayerNorm(layers[2])
        
        self.layer_4 = nn.Linear(layers[2], layers[3])
        self.batch_norm_4 = nn.LayerNorm(layers[3])
        
        self.layer_5 = nn.Linear(layers[3], 1)
         
        # init weights
        self.reset_weights()
    
    def reset_weights(self):
        
        # Инициализация весов для слоев
        for layer in [self.layer_1, self.layer_2, self.layer_3, self.layer_4]:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
            
        # Инициализация выходного слоя
        self.layer_5.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, state):
        x = self.relu(self.batch_norm_1(self.layer_1(state)))
        
        v = self.relu(self.batch_norm_2(self.layer_2(x)))
        v = self.relu(self.batch_norm_3(self.layer_3(v)))
        v = self.relu(self.batch_norm_4(self.layer_4(v)))
        
        w = self.relu(self.batch_norm_2(self.layer_2(x)))
        w = self.relu(self.batch_norm_3(self.layer_3(w)))
        w = self.relu(self.batch_norm_4(self.layer_4(w)))
        
        action_v = self.tanh(self.layer_5(v))* self.action_max
        action_w = self.tanh(self.layer_5(w))* self.action_max
        
        action = torch.cat((action_v, action_w), dim=-1)
        return action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chekpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chekpoint))