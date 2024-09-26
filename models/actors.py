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
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        self.layer_2 = nn.Linear(layers[0],layers[1])
        self.batch_norm_2 = nn.BatchNorm1d(layers[1])
        self.layer_3 = nn.Linear(layers[1],layers[2])
        self.batch_norm_3 = nn.BatchNorm1d(layers[2])
        self.layer_4 = nn.Linear(layers[2],layers[3])
        self.batch_norm_4 = nn.BatchNorm1d(layers[3])
        self.layer_5 = nn.Linear(layers[3],action_size)
        # init weights
        self.reset_weights()
    
    def reset_weights(self):
        nn.init.kaiming_uniform_(self.layer_1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.layer_1.bias, 0.1)
        nn.init.kaiming_uniform_(self.layer_2.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.layer_2.bias, 0.1)
        nn.init.kaiming_uniform_(self.layer_3.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.layer_3.bias, 0.1)
        nn.init.kaiming_uniform_(self.layer_4.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.layer_4.bias, 0.1)
        self.layer_5.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = self.batch_norm_1(x)
        x = F.relu(self.layer_2(x))
        x = self.batch_norm_2(x)
        x = F.relu(self.layer_3(x))
        x = self.batch_norm_3(x)
        x = F.relu(self.layer_4(x))
        x = self.batch_norm_4(x)
        action = F.tanh(self.layer_5(x))
        return action * self.action_max
