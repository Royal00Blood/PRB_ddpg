import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (ACTION_, STATE_SIZE, SEED,
                      ACTION_SIZE,LAYER_A1)
torch._dynamo.config.suppress_errors = False
    
class Actor_1(nn.Module):
    """ Арихитектура модели Актера для сети DDPG принимает на вход:
        state_size - размерность вектора состояния 
        action_size- размерность вектора действия
        seed=SEED  - параметр "скорости" обучения
        action_max - максимальное значение действия по модулю
        layers     - вектор со значениями количества нейронов в каждом слое
    """
    def __init__(self, 
                 state_size=STATE_SIZE, 
                 action_size=ACTION_SIZE, 
                 seed=SEED,
                 action_max = ACTION_,
                 layers=LAYER_A1 ):
        super(Actor_1, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.action_max = action_max
        
        self.layer_1 = nn.Linear(state_size, layers[0])
        # self.batch_norm_1 = nn.LayerNorm(layers[0])
        self.layer_2 = nn.Linear(layers[0],layers[1])
        # self.batch_norm_2 = nn.LayerNorm(layers[1])
        self.layer_3 = nn.Linear(layers[1],action_size)
        
        self.reset_weights()
    
    def reset_weights(self):
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in',nonlinearity='relu')
        nn.init.constant_(self.layer_1.bias, 0.1)
        nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in',nonlinearity='relu')
        nn.init.constant_(self.layer_2.bias, 0.1)
        self.layer_3.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        # x = self.batch_norm_1(x)
        x = F.relu(self.layer_2(x))
        # x = self.batch_norm_2(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        action = F.tanh(self.layer_3(x))
        return (action * ACTION_)


class Actor2(nn.Module):
    def __init__(self, state_dim, action_dims,action_max,layer_sizes):
        super(Actor2, self).__init__()
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.action_max = action_max
        self.layer_sizes = layer_sizes
        
        # Общие слои
        self.layer_1 = nn.Linear(state_dim, layer_sizes[0])
        self.layer_2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        
        # Отдельные слои для каждого действия
        self.action_layers = nn.ModuleList([
            nn.Linear(layer_sizes[1], action_dim) for action_dim in action_dims
        ])
    def reset_weights(self):
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in',nonlinearity='relu')
        nn.init.xavier_uniform_(self.action_layers.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        actions = []
        for layer in self.action_layers:
            action = torch.tanh(layer(x))
            actions.append(action)
        
        return actions
