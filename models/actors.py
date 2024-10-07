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
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, seed=SEED, action_max = A_MAX, layers=L_A, dir_chekpoint=DIR_CHEKPOINT, name = 'chekpoint_actor' ):
        super(Actor, self).__init__()
        
        self.chekpoint = os.path.join(dir_chekpoint, name)
        self.seed = torch.manual_seed(seed)
        self.action_max = action_max
        # init layers
        
        self.layer_1 = nn.Linear(state_size, layers[0])
        self.batch_norm_1 = nn.LayerNorm(layers[0])
        
        self.layer_2v = nn.Linear(layers[0], layers[1])
        self.batch_norm_2v = nn.LayerNorm(layers[1])
        
        self.layer_3v = nn.Linear(layers[1], layers[2])
        self.batch_norm_3v = nn.LayerNorm(layers[2])
        
        self.layer_4v = nn.Linear(layers[2], layers[3])
        self.batch_norm_4v = nn.LayerNorm(layers[3])
        
        
        self.layer_2w = nn.Linear(layers[0], layers[1])
        self.batch_norm_2w = nn.LayerNorm(layers[1])
        
        self.layer_3w = nn.Linear(layers[1], layers[2])
        self.batch_norm_3w = nn.LayerNorm(layers[2])
        
        self.layer_4w = nn.Linear(layers[2], layers[3])
        self.batch_norm_4w = nn.LayerNorm(layers[3])
        
        self.layer_5 = nn.Linear(layers[3], 1)
         
        # init weights
        self.reset_weights()
    
    def reset_weights(self):
        # Инициализация весов для слоев
        for layer in [self.layer_1, self.layer_2v, self.layer_3v, self.layer_4v, self.layer_2w ,self.layer_3w,self.layer_4w  ]:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
        # Инициализация выходного слоя
        self.layer_5.weight.data.uniform_(-0.1, 0.1)
        
        
    def forward(self, state):
        x = self.layer_1(state)
        x = F.relu(self.batch_norm_1(x))
        
        v = self.layer_2v(x)
        v = F.relu(self.batch_norm_2v(v))
        
        v = self.layer_3v(v)
        v = F.relu(self.batch_norm_3v(v))
        
        v = self.layer_4v(v)
        v = F.relu(self.batch_norm_4v(v))
        
        w = self.layer_2v(x)
        w = F.relu(self.batch_norm_2v(w))
        
        w = self.layer_3v(w)
        w = F.relu(self.batch_norm_3v(w))
        
        w = self.layer_4v(w)
        w = F.relu(self.batch_norm_4v(v))
        
        vw = torch.cat((self.layer_5(v), self.layer_5(w)), dim=-1)
        
        action = F.tanh(vw)* self.action_max
        return action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chekpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chekpoint))