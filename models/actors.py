import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (A_MAX, S_SIZE, A_SIZE, SEED, L_A)
torch._dynamo.config.suppress_errors = False
    
class Actor(nn.Module):
    """ Арихитектура модели Актера для сети DDPG принимает на вход:
        state_size  - размерность вектора состояния 
        action_size - размерность вектора действия
        seed        - параметр "скорости" обучения
        action_max  - максимальное значение действия по модулю
        layers      - вектор со значениями количества нейронов в каждом слое
    """
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, seed=SEED, action_max = A_MAX, layers=L_A):
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.action_max = action_max
        # init layers
        
        self.layer_1 = nn.Linear(state_size, layers[0])
        
        self.layer_2v = nn.Linear(layers[0],layers[1])
        self.layer_3v = nn.Linear(layers[1],layers[2])
        self.layer_4v = nn.Linear(layers[2],layers[3])
        self.layer_5v = nn.Linear(layers[3],1)
        
        self.layer_2w = nn.Linear(layers[0],layers[1])
        self.layer_3w = nn.Linear(layers[1],layers[2])
        self.layer_4w = nn.Linear(layers[2],layers[3])
        self.layer_5w = nn.Linear(layers[3],1)
        
        self.batch_norm_1 = nn.LayerNorm(layers[0])
        self.batch_norm_2 = nn.LayerNorm(layers[1])
        self.batch_norm_3 = nn.LayerNorm(layers[2])
        self.batch_norm_4 = nn.LayerNorm(layers[3])
        
        
        # init weights
        self.reset_weights()
    
    def reset_weights(self):
        # Инициализация весов для слоев
        for layer in [self.layer_1, self.layer_2v, self.layer_3v, self.layer_4v, self.layer_5v, self.layer_2w, self.layer_3w, self.layer_4w, self.layer_5w]:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
        # Инициализация выходного слоя
        self.layer_5v.weight.data.uniform_(-0.5, 0.5)
        self.layer_5w.weight.data.uniform_(-0.5, 0.5)
        
    def forward(self, state):
        x = self.layer_1(state)
        x = F.relu(self.batch_norm_1(x))
        
        v = self.layer_2v(x)
        v = F.relu(self.batch_norm_2(v))
        v = self.layer_3v(v)
        v = F.relu(self.batch_norm_3(v))
        v = self.layer_4v(v)
        v = F.relu(self.batch_norm_4(v))
        v = F.tanh(self.layer_5v(v))* 0.3
        
        w = self.layer_2v(x)
        w = F.relu(self.batch_norm_2(w))
        w = self.layer_3v(w)
        w = F.relu(self.batch_norm_3(w))
        w = self.layer_4v(w)
        w = F.relu(self.batch_norm_4(w))
        w = F.tanh(self.layer_5v(w))* self.action_max
        
        action = torch.cat((v, w), dim=-1)
        return action
